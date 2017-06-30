# dss_crf

Setup
```
sudo python setup.py install
```  

Usage
```
cd examples
python dense_hsal.py im1.png anno1.png out1.png
```
im1.png -> source image  
anno1.png -> predicted saliency map (should be gray level)  
out1.png -> output (after CRF)  
