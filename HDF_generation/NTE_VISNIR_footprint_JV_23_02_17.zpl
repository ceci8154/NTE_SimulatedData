# First load the relevant NTE spectrograph model
# Last time run on NTE_spec_230216_finall_coll.zmx
# 
# NB! Script can not edit operands that are pickups!
# NB! Change all wavelengths, orders, etc. to non-pickup before running
# NB! Make sure fields are (0,0), (0,0.9), (0,-0.9), (0.9,0), (-0.9,0)

# Uncomment below to select VIS or NIR
mode$ = "NIR"
#mode$ = "VIS"
#mode$ = "UV"

nstep = 50 # Number of samples along order

IF mode$ $== "NIR"
  start_conf = 15	# Configuration mainly used for check-points
  end_conf = 19
  start_order = 3	# Orders used for table output
  end_order = 7
  pixsize = 0.018	# Pixel width mm
  cx = 0 #0.4		# Detector offset X [mm]
  cy = 0 #1.5		# Detector offset Y [mm]
  nxpix = 2048
  nypix = 2048
ENDIF

IF mode$ $== "VIS"
  start_conf = 20	# Configuration mainly used for check-points
  end_conf = 24
  start_order = 8	# Orders used for table output
  end_order = 15 
  pixsize = 0.0105 	# Pixel width mm
  cx = 0 		# Detector offset X [mm]
  cy = -4.0		# Detector offset Y [mm]
  nxpix = 4096
  nypix = 1024
ENDIF

IF mode$ $== "UV"
  start_conf = 25	# Configuration mainly used for check-points
  end_conf = 28
  start_order = 16	# Orders used for table output
  #end_order = 20
  end_order = 21 # There is actually some little throughput in this order as well
  pixsize = 0.013	# Pixel width mm
  cx = -0.5		# Detector offset X [mm]
  cy = -1.5		# Detector offset Y [mm]
  nxpix = 1024
  nypix = 1024
ENDIF


gratingline = 23	# Line number defining grating in Zemax table

fsr_scale = 1.9	# Factor to scan orders beyond FSR

# Full field 1.8x1.8mm in slit plane
# Slit dimensions given as fraction of full field
#slitlength = 1 # 1.8mm
slitlength = 0.0233 # for pinhole

slitwidth = 0.0233 # 0.5"
#slitwidth = 0.0372 # 0.8"
#slitwidth = 0.0461 # 1.0"
#slitwidth = 0.0555 # 1.2"
#slitwidth = 0.0694 # 1.5"
#slitwidth = 0.0788 # 1.7"
#slitwidth = 0.0927 # 2.0"
#slitwidth = 0.2317 # 5.0"

PRINT "# Mode: ", mode$
PRINT "# Start order: ", start_order, " End order: ", end_order
PRINT "# Slit height: ", FLDX(0), " mm"
PRINT
PRINT "# COL1: Wavelength [um]"
PRINT "# COL2,3: Slit center    0,  0 [mm]"
PRINT "# COL4,5: Slit top      -X,  0 [mm]"
PRINT "# COL6,7: Slit left      0, -Y [mm]"
PRINT "# COL8,9: Slit bottom,   X,  0 [mm]"
PRINT "# COL10,11: Slit rigth,  0,  Y [mm]"

imgsrf = NSUR()

PLOT NEW
PLOT TITLE, "NTE ", mode$ ," footprint"
PLOT TITLEX, "X position [mm]"
PLOT TITLEY, "Y position [mm]"
PLOT CHECK, 0.003


# Now trace along orders for output to table

SETCONFIG start_conf
wl_save = WAVL(1)
order_save = PAR2(gratingline)

lines_per_um = PAR1(gratingline)
PRINT "# ", lines_per_um*1000," lines per mm"
line_dist = 1./(1e6*lines_per_um) # Line distance in metres
blaze_deg = ABSO(PAR3(gratingline-1))
PRINT "# Blaze angle ", blaze_deg," degrees"
PRINT

blaze_rad = blaze_deg*3.14159265/180.

DECLARE pointxfld, DOUBLE, 2, nstep, 5
DECLARE pointyfld, DOUBLE, 2, nstep, 5
DECLARE pointx, DOUBLE, 1, nstep
DECLARE pointy, DOUBLE, 1, nstep

FOR order = start_order,end_order,1
  # Set order, param 2
  SETSURFACEPROPERTY gratingline, 10, order, 2
  
  # Hack for VIS arm
  IF order==8
  	fsr_scale=2.3
  ENDIF
  IF order==9
  	fsr_scale=2.5
  ENDIF
  IF order==10
  	fsr_scale=2.8
  ENDIF
  IF order==11
  	fsr_scale=3
  ENDIF
  IF order==12
  	fsr_scale=3.2
  ENDIF
  IF order==13
  	fsr_scale=3.5
  ENDIF
  IF order==14
  	fsr_scale=3.8
  ENDIF
  IF order==15
  	fsr_scale=4
  ENDIF
  
  lam_cen = 1e6*2*line_dist*SINE(blaze_rad) / order
  fsr_half = lam_cen / (2*order)
  ws = lam_cen - fsr_half * fsr_scale
  we = lam_cen + fsr_half * fsr_scale
 
  PRINT
  PRINT "# Order ", order, " WL range ",ws, " - ", we, " um"
  step = (we-ws)/(nstep-1)

  FOR count = 1, nstep, 1
    wl = ws + (count-1)*step
    WAVL 1 = wl
    UPDATE

    FOR h = 1,5,1
      IF h==1
        xfield =  0
        yfield =  0
      ENDIF
      IF h==2
        xfield =  -slitwidth
        yfield =  0
      ENDIF
      IF h==3
        xfield =  0
        yfield =  -slitlength
      ENDIF
      IF h==4
        xfield = slitwidth
        yfield = 0
      ENDIF
      IF h==5
        xfield =  0
        yfield =  slitlength
      ENDIF
      
      RAYTRACE xfield, yfield, 0, 0, 1
      pointxfld(count, h) = RAYX(imgsrf)
      pointyfld(count, h) = RAYY(imgsrf)
    NEXT
    FORMAT 5.11
    PRINT, wl, " ", pointxfld(count,1) , " ",pointyfld(count, 1), " ",pointxfld(count, 2), " ",pointyfld(count, 2), " ",pointxfld(count, 3), " ",pointyfld(count, 3), " ",pointxfld(count, 4), " ",pointyfld(count, 4), " ",pointxfld(count, 5), " ",pointyfld(count, 5)
  NEXT

  !colour
  !colour=order - start_order
  colour = start_conf - (start_order - order)
  !style 0: solid  1-4 various 
  style = 0 
  !Option 0:only lines  1:both  2:only points
  opt = 0
  FOR h = 1,3,1
    FOR count = 1, nstep, 1
      IF h==1
        pointx(count)=pointxfld(count,1)
        pointy(count)=pointyfld(count,1)
      ENDIF
      IF h==2
        pointx(count)=pointxfld(count,3)
        pointy(count)=pointyfld(count,3)
      ENDIF
      IF h==3
        pointx(count)=pointxfld(count,5)
        pointy(count)=pointyfld(count,5)
      ENDIF
    NEXT
    PLOT DATA, pointx, pointy, nstep, colour, style, opt
  NEXT    
NEXT

!Restore Zemax tables
WAVL 1 = wl_save
SETSURFACEPROPERTY gratingline, 10, order_save, 2


# Draw detector
DECLARE detx, DOUBLE, 1, 5
DECLARE dety, DOUBLE, 1, 5

detx(1)=cx+pixsize*nxpix/2
dety(1)=cy+pixsize*nypix/2

detx(2)=cx+pixsize*nxpix/2
dety(2)=cy-pixsize*nypix/2

detx(3)=cx-pixsize*nxpix/2
dety(3)=cy-pixsize*nypix/2

detx(4)=cx-pixsize*nxpix/2
dety(4)=cy+pixsize*nypix/2

detx(5)=detx(1)
dety(5)=dety(1)

PLOT DATA, detx, dety, 5, 1, 0, 0

PLOT GO
