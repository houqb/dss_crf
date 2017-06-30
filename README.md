# dss_crf

The original version of this code can be found [here](https://github.com/lucasb-eyer/pydensecrf.git).
Notice that please provide a link to this project as a footnote or a citation if you plan to use it.

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
