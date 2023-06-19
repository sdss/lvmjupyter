import astrometry
import astropy.wcs
import astropy.units as u
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LogNorm

import numpy as np

import os
from os.path import exists

from photutils.detection import DAOStarFinder
from photutils.detection import IRAFStarFinder

from astropy.io import fits
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord, Angle

from scipy.ndimage import median_filter


homedir = "" #os.path.expanduser("~")

solver = astrometry.Solver(
    astrometry.series_5200.index_files(
        cache_directory=homedir + "astrometry_cache",
        scales={4, 5, 6},
    )
)

solver4200 = astrometry.Solver(
    astrometry.series_4200.index_files(
        cache_directory=homedir + "astrometry_cache",
        scales={1, 2, 3,4,5,6,7,8,9,10,11,12, 13, 14, 15},
    )
)

def logoddcallback(logodds_list):
    if len(logodds_list)>10:
        return astrometry.Action.STOP
    if logodds_list[0]>120.:
        return astrometry.Action.STOP
    return astrometry.Action.CONTINUE

def starfind(img, brightest=30, use=None, verbose=False):
    
    if use is None:
        use = "daophot"
    use = use.lower()
    
    image, _header = get_image_data(img, use_header = False)
    if not isinstance(image, np.ndarray):
        raise TypeError("type not supported")
    
    if use == "daophot":
    
        # use median filter to get rid of random hot pixels
        image = median_filter(image,size=5)#gaussian_filter(image_orig,sigma=5)



        # background estimate to later help DAOphot discern sources from background
        bkg_sigma = mad_std(image)  

        # disable saturated pixels
        #image_orig[image_orig>60000] = float("nan")

        # set up source finder from DAOphot 
        # we only want the brightest 30 sources, hence the "brightest" parameter
        daofind = DAOStarFinder(fwhm=4, threshold=2 * bkg_sigma,brightest=brightest)

        # use source finder on "image" and retrieve the sources found
        sources = daofind(image)  
        
        stars = sources
    
    elif use == "internal":
        raise NotImplemented
    
    
   
    
        dic = {}
        dic["xcentroid"] = []
        dic["ycentroid"] = []
        dic["flux"] = []
    
    
    if verbose:
        import matplotlib.pyplot as plt
        
        lower = np.nanmedian(image) - np.nanstd(image)
        upper = np.nanmedian(image) + 3*np.nanstd(image)
        
        plt.imshow(image, origin="lower", vmin=lower, vmax=upper)
        plt.plot(sources["xcentroid"],sources["ycentroid"],"o",markerfacecolor="none",markeredgecolor="r",ms=20)
        plt.title(img)
        plt.show()
        
    return stars

def wcs_from_file(*args, **kwargs):
    return platesolve(*args, **kwargs)

def wcs_from_header_of_file(filename):
    hdul = fits.open(filename)
    header = hdul[0].header
    
    return astropy.wcs.WCS(header)

def wcs_from_header(header):
    return astropy.wcs.WCS(header)

def header_from_file(filename):
    hdul = fits.open(filename)
    header = hdul[0].header
    return header
    

def get_image_data(str_or_data, use_header=True):
    image = str_or_data
    header = None
    if isinstance(image, str) and exists(image):
        if "center" in image:
            is_center = True
        
        hdul = fits.open(image)
        image = hdul[0].data
        if use_header:
            header = hdul[0].header
        else:
            header = None
        hdul.close()
    elif isinstance(image, str) and not exists(image):
        raise FileNotFoundError("file \"" + image + "\" not found. typo?")
    return image, header
    
    
def solve_list( sources , position_hint = None, lower_bound = 0.09, upper_bound = 5., sip_order = 4, tune_up_logodds_threshold = np.log(1.e6), logodds_callback = logoddcallback , verbose=False, header= None, solver=solver):
    
    if header is not None:
        try:
            radius = 2
            position_hint = astrometry.PositionHint(
                    ra_deg= float(header["RA"]),
                    dec_deg=float(header["DEC"]),
                    radius_deg=radius,
            )
            if verbose:
                print(f"set position hint to ra:{header['RA']} and dec:{header['DEC']} with radius {radius} degree") 
        except:
            position_hint = None
    if verbose:
        print("solver is : ", solver)
    try:
        # assume astrometry 3.0.0
        solution = solver.solve(
            # using the sources we found and their x and y coordinates
            stars_xs=sources['xcentroid'],
            stars_ys=sources['ycentroid'],
            # giv the solver the previously defined hints
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=lower_bound,#arcsec_per_pixel-0.1,
                upper_arcsec_per_pixel=upper_bound,#arcsec_per_pixel+0.1
            ),
            # for now, we do not want to privde a position hind
            # can later be read out from the fits file and given to the solver
            # which usually speeds up the solving process
            position_hint=position_hint,
            # example:
            #position_hint=astrometry.PositionHint(
            #    ra_deg=57.4,
            #    dec_deg=24.15,
            #    radius_deg=5,
            #),
            #solve_id=None,
            #tune_up_logodds_threshold=None,#14.0, # None disables tune-up (SIP distortion)
            #output_logodds_threshold=21.0,
            #logodds_callback=lambda logodds_list: logodds_callback

            # we here pass the parameters for the solution to the solver
            # that we do not want to tune up distortion parameters or we do 
            # as given in the function parameters
            solution_parameters=astrometry.SolutionParameters(
                sip_order=sip_order,
                tune_up_logodds_threshold=tune_up_logodds_threshold,
                logodds_callback=logoddcallback,
            ),
        )
        #log.debug("Solve done")
    except TypeError:
        # if there is a TypeError: solve() got an unexpected keyword argument 'stars_xs'
        # this might be astrometry 4.0.0, which just wants "stars" as keyword
         solution = solver.solve(
            # using the sources we found and their x and y coordinates
            stars=np.array([sources['xcentroid'], sources['ycentroid']]).T,
            # giv the solver the previously defined hints
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=lower_bound,#arcsec_per_pixel-0.1,
                upper_arcsec_per_pixel=upper_bound,#arcsec_per_pixel+0.1
            ),
            # for now, we do not want to privde a position hind
            # can later be read out from the fits file and given to the solver
            # which usually speeds up the solving process
            position_hint=position_hint,
            # example:
            #position_hint=astrometry.PositionHint(
            #    ra_deg=57.4,
            #    dec_deg=24.15,
            #    radius_deg=5,
            #),
            #solve_id=None,
            #tune_up_logodds_threshold=None,#14.0, # None disables tune-up (SIP distortion)
            #output_logodds_threshold=21.0,
            #logodds_callback=lambda logodds_list: logodds_callback

            # we here pass the parameters for the solution to the solver
            # that we do not want to tune up distortion parameters or we do 
            # as given in the function parameters
            solution_parameters=astrometry.SolutionParameters(
                sip_order=sip_order,
                tune_up_logodds_threshold=tune_up_logodds_threshold,
                logodds_callback=logoddcallback,
            ),
        )
        

    # if there was a solution, return the best match as astropy.wcs.WCS object
    # otherwise return None
    if solution.has_match():
        if verbose:
            print(f"{solution.best_match().center_ra_deg=}")
            print(f"{solution.best_match().center_dec_deg=}")
            print(f"{solution.best_match().scale_arcsec_per_pixel=}")
       
        return astropy.wcs.WCS(solution.best_match().wcs_fields)
    else:
        if verbose:
            print("No Match found!")
        return None
    
    
def platesolve(image, is_center = False, tune_up_logodds_threshold=None, sip_order=0, verbose=False, use_header=True, daostarfinder=False):
    ''', 
    returns wcs obj for a image (given as fits file location or np.ndarray)
    
    Necessary Parameters:
    image (str or np.ndarray):  A string holdng the path to a fits file, which will be solved
                                or a np.ndarray of image data (which might have originated
                                from a fits file). 
    
    Optional Parameters:                           
    is_center (bool):		during lvm testing we used a center camera, which had twice the
    				resolution. if "is_center" is True, data will be binned (for
    				better s/n and source detection). Is automatically set to True
    				if the path-string contains "center".
    				
    tune_up_logodds_threshold
    	      (float or None):  tune_up_logodds_threshold as used for the astrometry.Solver
    	      			Default is None (required for turning off distorion polynomials)
    	      			
    sip_order (int):		order of the sip distortion polynomials used (default is 0). must be
    				0 to not use any distortion polynomials.
    		
    verbose (bool):		prints some extra information if set to True, defaults to False
    
    use_header (bool):  if True (default), a position hint is given to the solver based on the Fits header keywords RA and DEC
    
    
    returns:
    astropy.wcs.WCS solution if a solution was found or None if no sulution was found
    
    
    
    example call:
    
    you may have a fits file in your folder called "example.fits", try this:
    
    >>> myWCS = platesolve("test.fits")
    >>> print(myWCS)
    
    
    '''
   
    # check if "image" is string and the filename exists
    # if so, read data from fits file
    image, header = get_image_data(image, use_header=use_header)
        
    # if "image" is not a np.ndarray at this point, discontinue
    if not isinstance(image, np.ndarray):
        raise TypeError("type not supported")
        
    
    # if "is_center" is True, bin data as we are using the raw bayer data
    if is_center:
        image = (image[::2,::2].astype(float)+
              image[1::2,::2].astype(float)+
              image[::2,1::2].astype(float)+
              image[1::2,1::2].astype(float))/4
              
    if verbose:
        print("image size:",image.shape)
    
    sources = starfind(image, use="daophot")
    
    if sources is None:
        print("No sources found, therefor no match found!")
        return None


    # nice table of sources found
    # and a nice plot of the image and the sources found
    # if verbose is set to True
    if verbose:
        for col in sources.colnames:  
            sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)  
        
        fig,ax1 = plt.subplots(figsize=(8,4.5))

        ax1.imshow(image,norm=LogNorm(vmin=50, vmax=1000))
        ax1.plot(sources["xcentroid"],sources["ycentroid"],"o",markerfacecolor="none",markeredgecolor="r",ms=20)
        plt.show()
    
    # set up callback action for faster astrometry rtermination in case of a found match
    

    
    # pixel scale hints for the solver (in arcsec/pixel)
    lower_bound = 0.8
    upper_bound = 1.2   
    
    # be more generous for the center camera     
    if is_center:
        lower_bound=0.4#arcsec_per_pixel-0.1,
        upper_bound=2.2#arcsec_per_pixel+0.1,
        
        
    if use_header:
        try:
            position_hint = astrometry.PositionHint(
                    ra_deg= float(header["RA"]),
                    dec_deg=float(header["DEC"]),
                    radius_deg=2,
            )
        except:
            position_hint = None
        
    else:
        position_hint = None
        
        
    # set up solver
    try:
        # assume astrometry 3.0.0
        solution = solver.solve(
            # using the sources we found and their x and y coordinates
            stars_xs=sources['xcentroid'],
            stars_ys=sources['ycentroid'],
            # giv the solver the previously defined hints
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=lower_bound,#arcsec_per_pixel-0.1,
                upper_arcsec_per_pixel=upper_bound,#arcsec_per_pixel+0.1
            ),
            # for now, we do not want to privde a position hind
            # can later be read out from the fits file and given to the solver
            # which usually speeds up the solving process
            position_hint=position_hint,
            # example:
            #position_hint=astrometry.PositionHint(
            #    ra_deg=57.4,
            #    dec_deg=24.15,
            #    radius_deg=5,
            #),
            #solve_id=None,
            #tune_up_logodds_threshold=None,#14.0, # None disables tune-up (SIP distortion)
            #output_logodds_threshold=21.0,
            #logodds_callback=lambda logodds_list: logodds_callback

            # we here pass the parameters for the solution to the solver
            # that we do not want to tune up distortion parameters or we do 
            # as given in the function parameters
            solution_parameters=astrometry.SolutionParameters(
                sip_order=sip_order,
                tune_up_logodds_threshold=tune_up_logodds_threshold,
                logodds_callback=logoddcallback,
            ),
        )
        #log.debug("Solve done")
    except TypeError:
        # if there is a TypeError: solve() got an unexpected keyword argument 'stars_xs'
        # this might be astrometry 4.0.0, which just wants "stars" as keyword
         solution = solver.solve(
            # using the sources we found and their x and y coordinates
            stars=np.array([sources['xcentroid'], sources['ycentroid']]).T,
            # giv the solver the previously defined hints
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=lower_bound,#arcsec_per_pixel-0.1,
                upper_arcsec_per_pixel=upper_bound,#arcsec_per_pixel+0.1
            ),
            # for now, we do not want to privde a position hind
            # can later be read out from the fits file and given to the solver
            # which usually speeds up the solving process
            position_hint=position_hint,
            # example:
            #position_hint=astrometry.PositionHint(
            #    ra_deg=57.4,
            #    dec_deg=24.15,
            #    radius_deg=5,
            #),
            #solve_id=None,
            #tune_up_logodds_threshold=None,#14.0, # None disables tune-up (SIP distortion)
            #output_logodds_threshold=21.0,
            #logodds_callback=lambda logodds_list: logodds_callback

            # we here pass the parameters for the solution to the solver
            # that we do not want to tune up distortion parameters or we do 
            # as given in the function parameters
            solution_parameters=astrometry.SolutionParameters(
                sip_order=sip_order,
                tune_up_logodds_threshold=tune_up_logodds_threshold,
                logodds_callback=logoddcallback,
            ),
        )
        

    # if there was a solution, return the best match as astropy.wcs.WCS object
    # otherwise return None
    if solution.has_match():
        if verbose:
            print(f"{solution.best_match().center_ra_deg=}")
            print(f"{solution.best_match().center_dec_deg=}")
            print(f"{solution.best_match().scale_arcsec_per_pixel=}")
       
        return astropy.wcs.WCS(solution.best_match().wcs_fields)
    else:
        if verbose:
            print("No Match found!")
        return None