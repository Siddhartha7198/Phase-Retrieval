
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation


class PhaseRetrieval():
    """
    Class for reconstructing real-space images from Fourier magnitudes
    """

    def __init__(self, fourier_mags, real_space_guess=None, step=100):
        self.measured_mags = fourier_mags       
        self.shape = self.measured_mags.shape
        self.ctr = self.shape[-1]//2
        if real_space_guess is not None:
            self.real_space_guess = real_space_guess #gaussian_filter(real_space_guess,5)
        else:
            self.real_space_guess = np.random.random_sample(self.shape)
        self.step = step
    
    def padding(self,recon):
        raw_size = recon.shape[-1]
        size = 2*raw_size
        padded = np.zeros((size,size,size),dtype = recon.dtype)
            
        padded[int(0.5*raw_size):int(1.5*raw_size),int(0.5*raw_size):int(1.5*raw_size),int(0.5*raw_size):int(1.5*raw_size)] = recon
        
        return padded

    def crop(self,vol): 
        size = vol.shape[-1]//2
        cropped = np.zeros((size,size,size),dtype = vol.dtype)
            
        cropped = vol[int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size)] 
        
        return cropped

    def real_RMSE(self, guess_im,true_im):
        """
        MSE in phase

        Parameters
        ----------
        guess_im : np.ndarray
            current real-space estimate
        true_im : np.ndarray  
            Reference magnitudes in Real space

        Returns
        -------
        float
            Mean-squared error in the phase of guess against true image
        """

        abs_err = np.sum((guess_im - true_im)**2)
        norm = np.sum(true_im**2)
        return np.sqrt(abs_err/norm)
            
    def fourier_MSE(self, guess):
        """
        MSE in Fourier domain

        Parameters
        ----------
        guess : np.ndarray
            Fourier transform of current real-space estimate
        reference : np.ndarray
            Reference magnitudes in Fourier space

        Returns
        -------
        float
            Mean-squared error in the Fourier domain of guess against self.measured_mags
        """
        abs_err = np.sum((np.abs(guess) - np.abs(self.measured_mags))**2)
        norm = np.sum(np.abs(self.measured_mags)**2)
        return abs_err/norm
           
    def align(self, unaligned, ref, return_meta=False):
        """
        Aligns the array "unaligned" to the array "ref" using phase cross-correlation.
        """
        min_error = float("inf")
        best_shift = None
        best_rot = 0
        best_aligned = None
        
        # Generate all 24 unique 90° rotations in 3D
        rotations = []
        for axes in [(0, 1), (0, 2), (1, 2)]:  # rotation planes
            for k in range(4):
                rotated = np.rot90(unaligned, k=k, axes=axes)
                rotations.append(((axes, k), rotated))
        
        # Deduplicate rotations to avoid repeats
        unique_rots = {}
        for (axes, k), arr in rotations:
            key = arr.tobytes()
            if key not in unique_rots:
                unique_rots[key] = (axes, k, arr)

        # Try each unique rotation
        for (axes, k, rotated) in unique_rots.values():
            shift, error, _ = phase_cross_correlation(ref, rotated)
            if error < min_error:
                min_error = error
                best_shift = shift.astype(int)
                best_rot = (axes, k)
                best_aligned = rotated
            
        # Try 4 rotations
#         for i in range(4):
#             rotated = np.rot90(unaligned, k=i)
#             shift, error, _ = phase_cross_correlation(ref, rotated)
#     
#             if error < min_error:
#                 min_error = error
#                 best_shift = shift.astype(int)
#                 best_rot = i
#                 best_aligned = rotated
    
        if best_shift is not None:
            best_aligned = np.roll(best_aligned, best_shift, axis=(0, 1, 2))
    
        return best_aligned, best_rot, best_shift


            
    def calc_real_space_error(self, true_im, plot=False, return_meta=False):
        """
        Determines the proper rotation and translation for matching the reconstructed real space image to the true one

        inputs
        --------
        true_im : ndarry of the original real space image
        plot : boolean
            whether to plot the shifted image an and the final error plot.
        """


        fixed, n_rot, shift = self.align(self.real_space_guess,true_im,True)
        
        fixed = fixed[
            int(0.5*self.ctr):int(1.5*self.ctr),
            int(0.5*self.ctr):int(1.5*self.ctr),
            int(0.5*self.ctr):int(1.5*self.ctr)]
            
        if return_meta:
            
            true_im = true_im[
                int(0.5*self.ctr):int(1.5*self.ctr),
                int(0.5*self.ctr):int(1.5*self.ctr),
                int(0.5*self.ctr):int(1.5*self.ctr)]
            
            min_error = float("inf")
                                               
            for i in range(1,len(self.real_space_err_track)+1):

                temp = np.load(f'recon_{i*self.step}.npz')['recon']
                
                temp = np.roll(np.rot90(temp,k=n_rot[1],axes=n_rot[0]),(shift[0],shift[1],shift[2]),axis=(0,1,2))                    
                
                temp = temp[
                    int(0.5*self.ctr):int(1.5*self.ctr),
                    int(0.5*self.ctr):int(1.5*self.ctr),
                    int(0.5*self.ctr):int(1.5*self.ctr)]
                
                error = self.real_RMSE(temp, true_im) 
                self.real_space_err_track[i-1] = error
                
                ############# The best reconstruction over iterations #############
                if error < min_error: 
                    min_error = error
                    best = temp
        
                
        if plot:           
                   
            plt.plot(self.real_space_err_track)
            plt.yscale("log")
            # plt.ylim(top=1)
            plt.title('Phase Retrieval')
            plt.ylabel('Root Mean Squared Error')
            plt.xlabel('Iterations')
            
            plt.show()
            
  
        return (fixed, self.real_space_err_track, best) if return_meta else fixed

            
        
    def _step(self, density_mod_func, curr_iter, **kwargs):
        """
        One iteration of the hybrid input output (HIO) algorithm with given beta value

        Parameters
        ----------
        denisty_mod_func : callable
            Function to update pixel values.

        Returns
        -------
        fourier_err : float
            Mean squared error in fourier domain - see fourier_MSE above

        rs_non_density_modified : ndarray
            Updated real space guess without any density modificaiton applied

        new_real_space : nd_array
        """
        ft = np.fft.fftshift(np.fft.fftn(self.real_space_guess))
        
        #fourier_err = self.fourier_MSE(ft)   
        
        ##### Hole at center #####
        # hole = ft[self.ctr-5:self.ctr+6, 
        #           self.ctr-5:self.ctr+6, 
        #           self.ctr-5:self.ctr+6] 
        
        # self.measured_mags[self.ctr-5:self.ctr+6, 
        #                    self.ctr-5:self.ctr+6, 
        #                    self.ctr-5:self.ctr+6] = np.abs(hole)
                                                               

        # Mix known magnitudes and guessed phases
        ks_est = self.measured_mags*np.exp(1j*np.angle(ft))

        # Inverse fourier transfrom your phase guess with the given magnitudes
        rs_non_density_modified = np.real(np.fft.ifftn(np.fft.ifftshift(ks_est)))

        # Impose desired real-space density constraint
        # gamma  = np.real(rs_non_density_modified) > 0 # Mask of positive density
        # new_real_space = rs_non_density_modified*gamma - (rs_non_density_Modified*(~gamma)*beta)
        self.real_space_guess[:] = density_mod_func(rs_non_density_modified, self.real_space_guess, curr_iter, **kwargs)
                
        return self.real_space_guess

    def _initialize_tracking(self, n_iter):
        """
        Set up tracking arrays for an iterative algorithm.

        Parameters
        ----------
        n_iter : int
            Number of density modification steps to take.
        """

        self.real_space_err_track = np.zeros(n_iter//self.step)
        return

    def iterate(self, density_mod_func, n_iter,prog_bar =False, **kwargs):
        """
        Run iterations of phase retrieval algorithm specified by the
        density modification function
        """
        self._initialize_tracking(n_iter)

        for i in range(1,n_iter+1):
            print(f'Iteration: {i}/{n_iter}', flush=True)    
            recon = self._step(density_mod_func, i, **kwargs)
            
            if i % self.step == 0 or i == n_iter:                 
                np.savez_compressed(f'recon_{i}.npz', recon=recon)
                
        return

    def _ERupdate(self, density, old_density, curr_iter):
        return density*(density > 0)

    def _IOupdate(self, density, old_density, curr_iter, beta):
        gamma = density > 0
        return density*gamma - (~gamma*(old_density - (beta*density)))

    def _HIOupdate(self, density, old_density, curr_iter, beta, freq):
        # Input-Output
        if np.random.rand() < freq:
            return self._IOupdate(density, old_density, curr_iter, beta)
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def _CHIOupdate(self, density, old_density, curr_iter, alpha, beta, freq):
        gamma = density>alpha*old_density
        delta = (0<density)*(~gamma)
        negatives = ~(gamma+delta)
        # CHIO
        if np.random.rand() < freq:
            return density*gamma + delta*(old_density-((1-alpha)/alpha)*density) + (old_density - beta*density)*negatives
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def _BoundedCHIOupdate(self, density, old_density, curr_iter, alpha, beta, freq):
        gamma = density>alpha*old_density
        delta = (0<density)*(~gamma)
        negatives = ~(gamma+delta)
        # Bounded CHIO
        if np.random.rand() < freq:
            chio = density*gamma + delta*(old_density-((1-alpha)/alpha)*density) + (old_density - beta*density)*negatives
            return  chio*(np.abs(chio)<1) + (np.abs(chio)>1)
        # Error Reduction
        else:
            return self._ERupdate(density, old_density, curr_iter)

    def ErrorReduction(self, n_iter=None,**kwargs):
        """
        Implementation of the error reduction phase retrieval algorithm
        from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run error reduction for n_iter iterations
        self.iterate(self._ERupdate, n_iter,**kwargs)
        return

    def InputOutput(self, beta=0.9, n_iter=None,**kwargs):
        """
        Implementation of the input-output phase retrieval algorithm
        from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        n_iters : int
            Number of iterations to run algorithm
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run input-output for n_iter iterations
        self.iterate(self._IOupdate, n_iter, beta=beta,**kwargs)
        return

    def HIO(self, beta=0.9, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the hybrid input-output phase retrieval
        algorithm from Fienup JR, Optics Letters (1978).

        Parameters
        ----------
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run HIO for n_iter iterations
        self.iterate(self._HIOupdate, n_iter, beta=beta, freq=freq,**kwargs)
        return

    def CHIO(self, alpha=0.4, beta=0.9, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the continuous hybrid input-output phase
        retrieval algorithm

        Parameters
        ----------
        alpha : float
            Scaling coefficient for modifying small real-space density
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run CHIO for n_iter iterations
        self.iterate(self._CHIOupdate, n_iter, alpha=alpha, beta=beta,
                     freq=freq,**kwargs)
        return

    def BoundedCHIO(self, alpha=0.4, beta=0.9, freq=0.95, n_iter=None,**kwargs):
        """
        Implementation of the continuous hybrid input-output phase
        retrieval algorithm

        Parameters
        ----------
        alpha : float
            Scaling coefficient for modifying small real-space density
        beta : float
            Scaling coefficient for modifying negative real-space
            density
        freq : float
            Frequency with which to use input-output updates
        n_iters : int
            Number of iterations to run algorithm
        """
        if n_iter is None:
            raise ValueError("Number of iterations must be specified")

        # Run Bounded CHIO for n_iter iterations
        self.iterate(self._BoundedCHIOupdate, n_iter, alpha=alpha,
                     beta=beta, freq=freq,**kwargs)
        return

