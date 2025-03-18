# Toy Retrieval

Simple Jupyter Notebook toy retrieval where we generate synthetic radiances and then try to retrieve them.

## Dependencies

* absco_lookup.py: code to read the absorption coefficient file (absco.h5)
* find_nearest.py: simple function to find the index of the nearest value in an array
* mie.py: Bohren and Huffman Mie scattering theory
* retrieval.py: contains the majority of the forward model and retrieval code
* rt_simple.py: contains the simple radiative transfer model part of the forward model
* rt_xrtm.py: contains the XRTM radiative transfer model part of the forward model

## Setup

* You'll need to make sure ABSCO_TABLE_FOLDER in settings.py is pointing to the folder containing absco.h5
* You can also change the band ranges, spectral resolutions, geometry, SNR, etc. within settings.py

## Installing XRTM

* XRTM can be downloaded at: https://reef.atmos.colostate.edu/~gregm/xrtm/

* Documentation for building and using XRTM are located on the webpage.

* The XRTM Python interface module is "wherever/xrtm/interfaces/XRTM.so".  The directory wherever/xrtm/interfaces must be in your PYTHONPATH environmental variable.

## Executing program

* In a shell:

```
jupyter notebook toy_retrieval.ipynb
```

* Or with examples using XRTM and comparison with the simple RT model:

```
jupyter notebook toy_retrieval_xrtm.ipynb
```

* If you change settings.py, you may need to restart your ipynb kernel for it to see those changes
* You can also run the Jupyter Notebook in a GUI such as Visual Studio Code
