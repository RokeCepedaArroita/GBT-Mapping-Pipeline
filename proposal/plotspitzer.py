import numpy as np
import matplotlib.pyplot as plt
import datetime
from tools import *
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')

''' Script to plot Spitzer data in context to simulated data in order to show in proposal '''

# Quick Settings

spitzer_base = '/scratch/nas_falcon/scratch/rca/projects/gbt/data/b30_spitzer'
irac1a_map = 'r14533376/ch1/pbcd/SPITZER_I1_14533376_0000_5_E8686015_maic.fits'
irac2a_map = 'r14533376/ch2/pbcd/SPITZER_I2_14533376_0000_5_E8675306_maic.fits'
irac3a_map = 'r14533376/ch3/pbcd/SPITZER_I3_14533376_0000_5_E8676220_maic.fits'
irac4a_map = 'r14533376/ch4/pbcd/SPITZER_I4_14533376_0000_5_E8689046_maic.fits'
irac1b_map = 'r14533632/ch1/pbcd/SPITZER_I1_14533632_0672_4_E8618115_maic.fits'
irac2b_map = 'r14533632/ch2/pbcd/SPITZER_I2_14533632_0672_4_E8611846_maic.fits'
irac3b_map = 'r14533632/ch3/pbcd/SPITZER_I3_14533632_0672_4_E8611741_maic.fits'
irac4b_map = 'r14533632/ch4/pbcd/SPITZER_I4_14533632_0672_4_E8617861_maic.fits'
mips1_map = 'r14535936/ch1/pbcd/SPITZER_M1_14535936_0000_7_E6323795_maic.fits'
mips2_map = 'r14535936/ch2/pbcd/SPITZER_M2_14535936_0000_7_E6322976_maic.fits'
mips3_map = 'r14535936/ch3/pbcd/SPITZER_M3_14535936_0000_7_E6322646_maic.fits'

irac_bands = [3.550, 4.493, 5.731, 7.872]
mips_bands = [23.68, 71.42, 155.9]


molecular_clouds = {'locs': [[192.1572,-11.1062], [192.3958,-11.3262], [192.1106,-11.7302], [192.3917,-12.2417], [192.5388,-11.5275], [192.3249,-12.0295], [192.3156,-10.7656], [191.3483,-11.0403], [192.2851,-09.0423], [192.7880,-09.0563], [191.9490, -11.5458]],
                   'names': ['LDN 1582B','LDN 1582','LDN 1577','LDN 1583','LDN 1584','LDN 1581','LDN 1580','LDN 1573','LDN 1579','LDN 1585', 'B30']}

central_coords = [192.41, -11.51]


contour_data = '/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_13.7GHz_mK.fits'


# # Initialise plot
#
# image_cband, nhits_cband, rms_cband, wcs_cband = open_simulation('/scratch/nas_falcon/scratch/rca/projects/gbt/sdosim/gbt_outputs', 'daisy_repeats000_4.85GHz.fits')
#
# plt.figure(1,figsize=(8,8))
# ax = plt.subplot(projection=wcs_cband)
# plt.imshow(image_cband*1, cmap='binary')
#
#
# # Print contours
# arrayhalpha, footprinthalpha = healpix2wcs(data_location='/scratch/nas_falcon/scratch/rca/projects/gbt/data/ancillary_data/Halpha_fwhm06_1024.fits', target_wcs=wcs_cband, shape=np.shape(image_cband))
# arraydust, footprintdust = healpix2wcs(data_location='/scratch/nas_falcon/scratch/rca/projects/gbt/data/ancillary_data/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits', target_wcs=wcs_cband, shape=np.shape(image_cband))
#
# plt.imshow(arrayhalpha, cmap='Reds')
# plt.imshow(arraydust, cmap='Blues', alpha=0.7)
# #plt.imshow(arrayhalpha, cmap='reds', alpha=0.5)
# # ax.contour(arraydust, colors='blue', levels=5, alpha=0.8, linewidths=0.5)
# # ax.contour(arrayhalpha, colors='red', levels=5, alpha=0.8, linewidths=0.5)
#
#
# # Print spitzer fields: irac A, mips 24
# # irac in green, mips in orange both thick lines
#
# # Contour plot of a boolean image of 1s and 0s, for now only mips 1 and irac 1
#
# array_edges, footprint = reproject_edges(f'/scratch/nas_falcon/scratch/rca/projects/gbt/COMAP/fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band6-RCACROP.fits', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#F6AE2D', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{mips1_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='g', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac4b_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#D84A05', levels=[0.000001], alpha=0.8, linewidths=1)
#
#
#
#
# '''
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{mips2_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='g', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{mips3_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='g', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac1a_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac2a_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac3a_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac4a_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac1b_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac2b_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# array_edges, footprint = reproject_edges(f'{spitzer_base}/{irac3b_map}', wcs_cband, shape=np.multiply(np.shape(image_cband),20))
# ax.contour(array_edges, colors='#FFA500', levels=[0.000001], alpha=0.8, linewidths=1)
#
# '''
#
# # First have a coverage figure
# # Print C-band reach as circle
# circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs_cband, center=central_coords, radius=1.5, npoints=360)
# ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)
#
# # Print Ku-band reach as circle
# circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs_cband, center=central_coords, radius=0.4, npoints=360)
# ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)
#
# # Plot locations of clouds
# if not molecular_clouds is None:
#     for cloud_loc in molecular_clouds['locs']:
#         pix_x, pix_y = wcs_cband.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
#         ax.plot(pix_x, pix_y, '^', color='k')
#
# # Mark centre of observation
# pix_x1, pix_y1 = wcs_cband.wcs_world2pix(central_coords[0], central_coords[1], 0)
# ax.plot(pix_x1, pix_y1, 'x', color='k')
#
#
# # Lambda orionis position
# point_at_radius = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=1.1, wcs=wcs_cband)
# point_at_radius2 = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=1.35, wcs=wcs_cband)
# delta_point_radius = np.subtract(point_at_radius2, point_at_radius)
# ax.text(point_at_radius[0], point_at_radius[1]+3.5, r'$\lambda$ Ori', color='#878787', rotation=np.rad2deg(np.arctan(delta_point_radius[1]/delta_point_radius[0])))
# plt.arrow(point_at_radius[0], point_at_radius[1], delta_point_radius[0], delta_point_radius[1], head_width=2.5, head_length=6, fc='#878787', ec='#878787', linewidth=1.5)
#
# ''' All wcs world2pix conversions should have 0 at the end since the first pixel is 0,0. I checked by plotting '''
#
# # Set limits
# pix_x1, pix_y1 = wcs_cband.wcs_world2pix(central_coords[0]-1.5, central_coords[1]-1.5, 0)
# pix_x2, pix_y2 = wcs_cband.wcs_world2pix(central_coords[0]+1.5, central_coords[1]+1.5, 0)
# ax.set_xlim(np.sort([pix_x1,pix_x2]))
# ax.set_ylim(np.sort([pix_y1,pix_y2]))
#
# #plt.title('Observation Coverage Map', y=1.015, fontsize=11)
# plt.xlabel('Galactic Longitude')
# plt.ylabel('Galactic Latitude')
#
# plt.savefig(f'./coverage_map.png', bbox_inches='tight', pad_inches=0, dpi=600)
# plt.savefig(f'./coverage_map.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
#
# # Show figure
# plt.show()


#
# for i,mapname in enumerate([irac4a_map]): #enumerate([irac1a_map, irac2a_map, irac3a_map, irac4a_map]):
#
#     i=3
#     titlename = f'IRAC Ch{i+1} (A): {irac_bands[i]}' + r'$\,\mathrm{\mu m}$'
#
#     # Open data
#     image, wcs = open_spitzer(spitzer_base, mapname)
#     #image, wcs = open_comap('/scratch/nas_falcon/scratch/rca/projects/gbt/COMAP/fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band6-RCACROP.fits')
#
#     # Reproject contour data
#     array, footprint = healpix2wcs(data_location=contour_data, target_wcs=wcs, shape=np.shape(image))
#
#
#     def plot_and_save(image, wcs):
#
#         plt.figure(1)
#         ax = plt.subplot(projection=wcs)
#
#         plt.imshow(image, vmin=0, vmax=np.nanmedian(image)*2)
#
#         print(np.shape(image))
#
#         centralx, centraly = wcs.wcs_pix2world(np.round(np.shape(image)[1]/2), np.round(np.shape(image)[0]/2), 0)
#         corner1x, corner1y = wcs.wcs_pix2world(0, 0, 0)
#         corner2x, corner2y = wcs.wcs_pix2world(np.round(np.shape(image)[1]), np.round(np.shape(image)[0]), 0)
#
#
#         print('center', centralx, centraly)
#         print('corner 1', corner1x, corner1y)
#         print('corner 2', corner2x, corner2y)
#         print('delta lon', corner1x-corner2x)
#         print('delta lat', corner1y-corner2y)
#         print(f'Total area is {np.abs( (corner1y-corner2y)*(corner1x-corner2x) )* 0.978} sq deg')
#
#
#         # Find real corners
#         good_pix = np.where(~np.isnan(image))
#         x_range = [np.nanmin(good_pix[0][good_pix[0]==np.nanmin(good_pix[0])]), np.nanmax(good_pix[0][good_pix[0]==np.nanmax(good_pix[0])])]
#         y_range = [np.nanmin(good_pix[1][good_pix[1]==np.nanmin(good_pix[1])]), np.nanmax(good_pix[1][good_pix[1]==np.nanmax(good_pix[1])])]
#
#         realcorner1x, realcorner1y = wcs.wcs_pix2world(x_range[0], y_range[0], 0)
#         realcorner2x, realcorner2y = wcs.wcs_pix2world(x_range[1], y_range[1], 0)
#         print('realcorner1', realcorner1x, realcorner1y)
#         print('realcorner2', realcorner2x, realcorner2y)
#
#
#         print(x_range, y_range)
#
#
#         #plt.axes(projection=wcs)
#
#         ax.set_facecolor(('#a9a9a9'))
#         #ax.set_xticks([12,12.5])
#
#
#         # Grid
#         plt.grid(color='white', ls='solid', linewidth=0.5)
#
#         # Plot contours
#         ax.contour(array, colors='black', alpha=0.8, linewidths=0.5)
#
#         # Colorbar
#         #cbar = plt.colorbar()
#         #cbar.set_label(r'$\mathrm{MJy sr^{-1}}$', labelpad=0.8)
#
#         # Plot locations of clouds
#         from astropy import units as u
#         from astropy.coordinates import SkyCoord
#         for cloud_loc in molecular_clouds['locs']:
#             c = SkyCoord(l=cloud_loc[0]*u.degree, b=cloud_loc[1]*u.degree, frame='galactic')
#             print(c.icrs.ra.value, c.icrs.dec.value)
#             pix_x, pix_y = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
#             ax.plot(pix_x, pix_y, '^', color='k')
#
#         # Mark centre of observation
#         c = SkyCoord(l=central_coords[0]*u.degree, b=central_coords[1]*u.degree, frame='galactic')
#         pix_x1, pix_y1 = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
#         ax.plot(pix_x1, pix_y1, 'x', color='k')
#
#         # Plot circle around Ku observations
#         circle_x, circle_y = circle_in_radecworldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
#         ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)
#
#         # Labels
#         plt.ylabel('Right Ascension')
#         plt.xlabel('Declination')
#         plt.title(titlename)
#
#         # Set limits
#         pix_x1, pix_y1 = wcs.wcs_world2pix(corner1x, corner1y, 0)
#         pix_x2, pix_y2 = wcs.wcs_world2pix(corner2x, corner2y, 0)
#         ax.set_xlim(np.sort([pix_x1,pix_x2]))
#         ax.set_ylim(np.sort([pix_y1,pix_y2]))
#
#
#         lon, lat = ax.coords
#         lon.set_ticks(color='k')
#         lon.set_ticks_position('lb')
#         lon.set_ticklabel_position('lb')
#         lat.set_ticks(color='k')
#         lat.set_ticks_position('lb')
#         lat.set_ticklabel_position('lb')
#
#
#
#         plt.show()
#         plt.savefig(f'./irac_A_ch{i+1}.png', bbox_inches='tight', pad_inches=0, dpi=600)
#         plt.savefig(f'./irac_A_ch{i+1}.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
#         #plt.show()
#
#     # Plot data
#
#     plot_and_save(image, wcs)
#
#







for i,mapname in enumerate([irac4b_map]): #enumerate([irac1b_map, irac2b_map, irac3b_map, irac4b_map]):

    i = 3

    titlename = f'IRAC Ch{i+1}: {irac_bands[i]}' + r'$\mathrm{\mu m}$'

    # Open data
    image, wcs = open_spitzer(spitzer_base, mapname)

    # Reproject contour data
    array, footprint = healpix2wcs(data_location=contour_data, target_wcs=wcs, shape=np.shape(image))


    def plot_and_save(image, wcs):

        plt.figure(1)
        ax = plt.subplot(projection=wcs)

        plt.imshow(image, vmin=0, vmax=np.nanmedian(image)*1.8)


        print(np.shape(image))

        centralx, centraly = wcs.wcs_pix2world(np.round(np.shape(image)[1]/2), np.round(np.shape(image)[0]/2), 0)
        corner1x, corner1y = wcs.wcs_pix2world(0, 0, 0)
        corner2x, corner2y = wcs.wcs_pix2world(np.round(np.shape(image)[1]), np.round(np.shape(image)[0]), 0)


        print('center', centralx, centraly)
        print('corner 1', corner1x, corner1y)
        print('corner 2', corner2x, corner2y)
        print('delta lon', corner1x-corner2x)
        print('delta lat', corner1y-corner2y)
        print(f'Total area is {np.abs( (corner1y-corner2y)*(corner1x-corner2x) )* 0.978} sq deg')


        ax.set_facecolor(('#a9a9a9'))


        # Grid
        plt.grid(color='white', ls='solid', linewidth=0.5)

        # Plot contours
        ax.contour(array, colors='black', alpha=0.7, linewidths=0.5)

        # Colorbar
        #cbar = plt.colorbar()
        #cbar.set_label(r'$\,\mathrm{MJy sr^{-1}}$', labelpad=0.8)

        # Plot locations of clouds
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        for cloud_loc in molecular_clouds['locs']:
            c = SkyCoord(l=cloud_loc[0]*u.degree, b=cloud_loc[1]*u.degree, frame='galactic')
            print(c.icrs.ra.value, c.icrs.dec.value)
            pix_x, pix_y = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
            ax.plot(pix_x, pix_y, '^', color='k')

        # Mark centre of observation
        c = SkyCoord(l=central_coords[0]*u.degree, b=central_coords[1]*u.degree, frame='galactic')
        pix_x1, pix_y1 = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
        ax.plot(pix_x1, pix_y1, 'x', color='k')

        # Plot circle around Ku observations
        circle_x, circle_y = circle_in_radecworldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
        ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)


        # Lambda orionis position
        point_at_radius = straight_line_radius_calc_radec(central_coords,[195.05189264, -11.99506278],radius=0.35, wcs=wcs)
        point_at_radius2 = straight_line_radius_calc_radec(central_coords,[195.05189264, -11.99506278],radius=0.47, wcs=wcs)
        delta_point_radius = np.subtract(point_at_radius2, point_at_radius)
        ax.text(point_at_radius[0]-630, point_at_radius[1]+330, r'$\lambda$ Ori', color='#c3c3c3', fontsize=12, rotation=np.rad2deg(np.arctan(delta_point_radius[1]/delta_point_radius[0])))
        plt.arrow(point_at_radius[0], point_at_radius[1], delta_point_radius[0], delta_point_radius[1], head_width=100, head_length=150, fc='#c3c3c3', ec='#c3c3c3', linewidth=1.5)

        # Set limits
        corner1x = 82.47
        corner2x = 83.20
        corner1y = 11.85
        corner2y = 12.95

        # Labels
        plt.ylabel('Right Ascension')
        plt.xlabel('Declination')
        plt.title(titlename)

        # Set limits
        pix_x1, pix_y1 = wcs.wcs_world2pix(corner1x, corner1y, 0)
        pix_x2, pix_y2 = wcs.wcs_world2pix(corner2x, corner2y, 0)
        ax.set_xlim(np.sort([pix_x1,pix_x2]))
        ax.set_ylim(np.sort([pix_y1,pix_y2]))

        lon, lat = ax.coords
        lon.set_ticks(color='k')
        lon.set_ticks_position('lb')
        lon.set_ticklabel_position('lb')
        lat.set_ticks(color='k')
        lat.set_ticks_position('lb')
        lat.set_ticklabel_position('lb')


        plt.show()
        plt.savefig(f'./irac_B_ch{i+1}.png', bbox_inches='tight', pad_inches=0, dpi=600)
        plt.savefig(f'./irac_B_ch{i+1}.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
        #

    # Plot data

    plot_and_save(image, wcs)




# mips_limits = [45, 50, 200]
#
# for i,mapname in enumerate([mips1_map, mips2_map, mips3_map]):
#
#     titlename = f'MIPS Ch{i+1}: {mips_bands[i]}' + r'$\,\mathrm{\mu m}$'
#
#     # Open data
#     image, wcs = open_spitzer(spitzer_base, mapname)
#
#     # Reproject contour data
#     contour_array, footprint = healpix2wcs(data_location=contour_data, target_wcs=wcs, shape=np.shape(image))
#
#
#     def plot_and_save(image, wcs, i):
#
#         plt.figure(1)
#         ax = plt.subplot(projection=wcs)
#         plt.imshow(image, vmin=np.nanmin(image), vmax=mips_limits[i])
#
#         centralx, centraly = wcs.wcs_pix2world(np.round(np.shape(image)[1]/2), np.round(np.shape(image)[0]/2), 0)
#         corner1x, corner1y = wcs.wcs_pix2world(0, 0, 0)
#         corner2x, corner2y = wcs.wcs_pix2world(np.round(np.shape(image)[1]), np.round(np.shape(image)[0]), 0)
#
#         print('center', centralx, centraly)
#         print('corner 1', corner1x, corner1y)
#         print('corner 2', corner2x, corner2y)
#         print('delta lon', corner1x-corner2x)
#         print('delta lat', corner1y-corner2y)
#         print(f'Total area is {np.abs( (corner1y-corner2y)*(corner1x-corner2x) )* 0.978} sq deg')
#
#
#         #plt.axes(projection=wcs)
#
#         ax.set_facecolor(('#a9a9a9'))
#         #ax.set_xticks([12,12.5])
#
#
#         # Grid
#         plt.grid(color='white', ls='solid', linewidth=0.5)
#
#         # Plot contours
#         ax.contour(contour_array, colors='black', alpha=0.7, linewidths=0.5)
#
#         # Colorbar
#         #cbar = plt.colorbar()
#         #cbar.set_label(r'$\mathrm{MJy sr^{-1}}$', labelpad=0.8)
#
#         # Plot locations of clouds
#         from astropy import units as u
#         from astropy.coordinates import SkyCoord
#         for cloud_loc in molecular_clouds['locs']:
#             c = SkyCoord(l=cloud_loc[0]*u.degree, b=cloud_loc[1]*u.degree, frame='galactic')
#             print(c.icrs.ra.value, c.icrs.dec.value)
#             pix_x, pix_y = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
#             ax.plot(pix_x, pix_y, '^', color='k')
#
#         # Mark centre of observation
#         c = SkyCoord(l=central_coords[0]*u.degree, b=central_coords[1]*u.degree, frame='galactic')
#         pix_x1, pix_y1 = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
#         ax.plot(pix_x1, pix_y1, 'x', color='k')
#
#         # Plot circle around Ku observations
#         circle_x, circle_y = circle_in_radecworldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
#         ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)
#
#         # Labels
#         plt.xlabel('Right Ascension')
#         plt.ylabel('Declination')
#         plt.title(titlename)
#
#
#         # Lambda orionis position
#         point_at_radius = straight_line_radius_calc_radec(central_coords,[195.05189264, -11.99506278],radius=0.35, wcs=wcs)
#         point_at_radius2 = straight_line_radius_calc_radec(central_coords,[195.05189264, -11.99506278],radius=0.47, wcs=wcs)
#         delta_point_radius = np.subtract(point_at_radius2, point_at_radius)
#         ax.text(point_at_radius[0]+30, point_at_radius[1]+130, r'$\lambda$ Ori', color='#c3c3c3', fontsize=12, rotation=np.rad2deg(np.arctan(delta_point_radius[1]/delta_point_radius[0])))
#         plt.arrow(point_at_radius[0], point_at_radius[1], delta_point_radius[0], delta_point_radius[1], head_width=20, head_length=30, fc='#c3c3c3', ec='#c3c3c3', linewidth=1.5)
#
#         # Set limits
#         corner1x = 82.07
#         corner2x = 82.97
#         corner1y = 11.75
#         corner2y = 12.95
#
#
#         pix_x1, pix_y1 = wcs.wcs_world2pix(corner1x, corner1y, 0)
#         pix_x2, pix_y2 = wcs.wcs_world2pix(corner2x, corner2y, 0)
#         ax.set_xlim(np.sort([pix_x1,pix_x2]))
#         ax.set_ylim(np.sort([pix_y1,pix_y2]))
#
#         plt.savefig(f'./mips_ch{i+1}.png', bbox_inches='tight', pad_inches=0, dpi=600)
#         plt.savefig(f'./mips_ch{i+1}.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
#         #plt.show()
#
#
#     # Plot data
#
#     plot_and_save(image, wcs, i)
