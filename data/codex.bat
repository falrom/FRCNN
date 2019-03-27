@echo off

set /a QP=22
set /a fps=50
set fileName=videos/BasketballDrive_1920x1080_50_000to049

echo File: %fileName%.yuv 
echo QP  : %QP%
echo FPS : %fps%
echo %date% %time%

echo encode...
:: del encode.log
x265.exe --input-res 1920x1080 --fps %fps% %fileName%.yuv -o %fileName%_QP%QP%_IP.bin --qp %QP% --ipratio 1 --bframes 0 --psnr --ssim 1>>encode.log 2>&1

echo decode...
:: del decode.log
TAppDecoder.exe -b %fileName%_QP%QP%_IP.bin -o %fileName%_QP%QP%_IP_rec.yuv 1>>decode.log 2>&1
