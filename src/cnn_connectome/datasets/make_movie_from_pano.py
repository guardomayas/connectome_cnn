import os,glob
import numpy as np
import h5py

import os,glob
import numpy as np
import h5py

class GetNaturalMovies:
    """
        Attributes
        ----------
        batch_imgs : np.ndarray
            Loaded and preprocessed images. Shape: (N, H, W).

        vel_traces : np.ndarray
            Velocity traces for each image and trace repetition.
            Shape: (N, traces_per_img, T).

        positions : np.ndarray
            Position (integrated velocity) traces.
            Shape: (N, traces_per_img, T).

        phase_indices : np.ndarray
            Precomputed cyclic pixel shifts for each phase.
            Shape: (phases_per_img, W).

        movies : np.ndarray
            Index table of all movie combinations.
            Shape: (num_movies, 3) containing (img_idx, trace_idx, phase_idx).

        all_movies : np.ndarray
            Rendered movies. Shape: (num_movies, T, H, fov).
    """
    def __init__(self, data_path='pano_scenes/', 
                 batch_idx=0, 
                 batch_size=5,
                 halfLife=0.2, 
                 velStd=100, 
                 sampleFreq=60, 
                 totalTime=3,
                 traces_per_img=2, 
                 phases_per_img=3, 
                 fov=250, 
                 percentile_scale = 99):
        self.data_path = data_path
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.traces_per_img = traces_per_img
        self.phases_per_img = phases_per_img
        self.halfLife = halfLife
        self.velStd = velStd
        self.sampleFreq = sampleFreq
        self.totalTime = totalTime
        self.numTime = int(sampleFreq * totalTime) + 1
        self.percentile_scale = percentile_scale
        self.fov = fov

        # Load images
        self.batch_imgs, self.names = self._load_batch()
        self.n_imgs, self.H, self.W = self.batch_imgs.shape

        # Initialize placeholders
        self.vel_traces = None
        self.positions = None
        self.phase_indices = None
        self.movies = None
        self.all_movies = None

        # Run pipeline
        self._generate_velocity_traces()
        self._generate_phase_indices()
        self._enumerate_movies()
        self._render_all_movies()

    ### METHOD DEFINITIONS ###
    # ------------------------------------------------------------------
    # Load and preprocess images
    # ------------------------------------------------------------------
    def _load_batch(self):
        mat_files = sorted(glob.glob(os.path.join(self.data_path, "*.mat")))
        start = self.batch_idx * self.batch_size
        end = start + self.batch_size
        batch_files = mat_files[start:end]

        imgs, names = [], []

        for fname in batch_files:
            with h5py.File(fname, 'r') as f:
                proj = np.array(f["projection"]).T
            names.append(os.path.basename(fname))

            imgs.append(proj)

        imgs = np.stack(imgs)
        scene_scale = np.percentile(imgs, self.percentile_scale, axis=(1, 2))
        imgs /= scene_scale[:, None, None]
        return np.clip(imgs, 0, 1), names


    # ------------------------------------------------------------------
    # Velocity traces
    # ------------------------------------------------------------------
    # Seed root for velocity-trace RNGs, kept distinct from
    # `_PHASE_SEED_ROOT` so the two randomization streams can never
    # collide. See `_generate_phase_indices` for why global (not
    # batch-local) indexing matters: the old scheme (`seed=i*100+tr`) used
    # the batch-local image index `i`, so e.g. local image 0 of batch 0 and
    # local image 0 of batch 1 drew identical velocity traces. Global
    # indexing makes each image's traces independent of how the corpus is
    # split into batches.
    _VEL_SEED_ROOT = 20260818_1

    def _generate_velocity_traces(self):
        self.vel_traces = np.zeros((self.n_imgs, self.traces_per_img, self.numTime))
        self.positions = np.zeros((self.n_imgs, self.traces_per_img, self.numTime))

        for i in range(self.n_imgs):
            global_idx = self.batch_idx * self.batch_size + i
            for tr in range(self.traces_per_img):
                vel = self._vel_trace(seed=(self._VEL_SEED_ROOT, global_idx, tr))
                pos = np.cumsum(vel) / self.sampleFreq
                pos -= pos[0]

                self.vel_traces[i, tr] = vel
                self.positions[i, tr] = pos

        print(f"[Vel] vel_traces: {self.vel_traces.shape}")
        print(f"[Vel] positions:  {self.positions.shape}")
    def _vel_trace(self, seed):
        T = self.numTime
        dt = 1.0 / self.sampleFreq
        tau = self.halfLife / np.log(2)
        alpha = np.exp(-dt / tau)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self.velStd, size=T)

        vel = np.zeros(T)
        for t in range(1, T):
            vel[t] = alpha * vel[t-1] + np.sqrt(1 - alpha**2) * noise[t]

        return vel
    # ------------------------------------------------------------------
    # Phase indices
    # ------------------------------------------------------------------
    # Seed root for phase-offset RNGs. Combined with a *global* image index
    # (not the batch-local index) via a SeedSequence, so offsets are both
    # independent per image and reproducible across different batch_idx /
    # batch_size choices. Kept as a distinct constant from the velocity
    # seeding scheme (`seed=i*100+tr` in `_generate_velocity_traces`) so the
    # two randomization streams can never accidentally collide.
    _PHASE_SEED_ROOT = 20260818

    def _generate_phase_indices(self):
        # Previously all images shared one set of offsets, drawn once from a
        # single global `self.rng`: phase index k always meant "shift by the
        # same k-th pixel offset" for every scene in the corpus. That ties
        # "phase index" to a fixed absolute azimuthal offset across the
        # whole dataset -- a confound for any analysis or split that groups
        # by phase. Each image now gets its own independent draw.
        self.phases_pixels = np.zeros((self.n_imgs, self.phases_per_img), dtype=int)
        self.phase_indices = np.zeros((self.n_imgs, self.phases_per_img, self.W), dtype=int)

        for i in range(self.n_imgs):
            global_idx = self.batch_idx * self.batch_size + i
            rng = np.random.default_rng((self._PHASE_SEED_ROOT, global_idx))
            phases_pixels = rng.integers(0, self.W, size=self.phases_per_img)

            self.phases_pixels[i] = phases_pixels
            self.phase_indices[i] = (
                np.arange(self.W)[None, :] - phases_pixels[:, None]
            ) % self.W

        print(f"[Phase] Phase indices: {self.phase_indices.shape}")
    # ------------------------------------------------------------------
    # Enumerate movies
    # ------------------------------------------------------------------
    def _enumerate_movies(self):
        self.movies = np.array(np.meshgrid(
            np.arange(self.n_imgs),
            np.arange(self.traces_per_img),
            np.arange(self.phases_per_img),
            indexing='ij'
        )).reshape(3, -1).T

        self.num_movies = self.movies.shape[0]
        print(f"[Movies] Total movies: {self.num_movies}")
    # ------------------------------------------------------------------
    # Render movies
    # ------------------------------------------------------------------
    def _render_all_movies(self):
        T, H, Ww = self.numTime, self.H, self.fov

        all_movies = np.empty((self.num_movies, T, H, Ww), dtype=np.float32)

        for m, (img_idx, trace_idx, phase_idx) in enumerate(self.movies):
            all_movies[m] = self._create_movie(
                self.batch_imgs[img_idx],
                self.positions[img_idx, trace_idx],
                phase_idx,
                self.phase_indices[img_idx],
                window_width=Ww
            )

        self.all_movies = all_movies
        print(f"[Render] all_movies: {self.all_movies.shape}")
    # ------------------------------------------------------------------
    # Create a single movie
    # ------------------------------------------------------------------
    def _create_movie(
            self,
            img,
            pos,
            phase_idx,
            phase_indices,
            window_width,
            ):
        shifted = img[:, phase_indices[phase_idx]]
        _, width = shifted.shape

        deg_per_pix = 360.0 / width

        # Continuous (sub-pixel) horizontal position, wrapped into [0, width).
        # Previously this was rounded to the nearest integer pixel, which
        # quantizes the motion trajectory: at low speeds especially, the FOV
        # would sit still for several frames and then jump a whole pixel,
        # injecting spurious high-frequency temporal structure that has
        # nothing to do with the true (smooth) OU velocity trace.
        pix_pos = (pos / deg_per_pix) % width

        pos_floor = np.floor(pix_pos).astype(int)
        frac = (pix_pos - pos_floor).astype(shifted.dtype)  # (T,) in [0, 1)

        offsets = np.arange(window_width) - window_width // 2

        idx0 = (pos_floor[:, None] + offsets[None, :]) % width  # (T, window_width)
        idx1 = (idx0 + 1) % width

        col0 = shifted[:, idx0]  # (H, T, window_width)
        col1 = shifted[:, idx1]  # (H, T, window_width)

        # Linear interpolation along the azimuthal (circular) axis only;
        # the elevation (H) axis is never resampled, so no 2D/bilinear
        # interpolation is needed there.
        movie = col0 * (1.0 - frac)[None, :, None] + col1 * frac[None, :, None]

        return movie.transpose(1, 0, 2)