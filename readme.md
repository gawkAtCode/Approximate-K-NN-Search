# ACM SIGMOD 2023 Programming Contest

This is the code for runner-up solution of [ACM SIGMOD 2023 Programming Contest](http://sigmod2023contest.eastus.cloudapp.azure.com/leaders.shtml). The main idea of this method can be found in the [poster](https://github.com/gawkAtCode/ACM-SIGMOD-2023-Programming-Contest-Runner-Up/blob/main/poster-CantonDwenDwen.pdf).

This code can generate an approximate K-nearest-neighbor graph in a very short time for tens of millions of high-dimensional vectors.

## Running environment 

Our program is written entirely in `python` and we write the dependency condition and library information of the program in [requirements.txt](https://github.com/gawkAtCode/ACM-SIGMOD-2023-Programming-Contest-Runner-Up/blob/main/requirements.txt).If you want to run the code successfully, please install these dependencies first.

Our program also uses [pynndescent](https://github.com/lmcinnes/pynndescent), but you do not need to download it. This is because we have made so many changes to the pyndescent source code that we have packaged it separately, in a folder named [pynndescent_opt](https://github.com/gawkAtCode/ACM-SIGMOD-2023-Programming-Contest-Runner-Up/tree/main/pynndescent_opt), where you can see our modified pyndescent code.

## Program introduction

Our program consists of two main parts,initial graph construction and graph refinement.
### Initial graph construction

After reading the content-data-release-10m.bin, we used faiss.index_factory(dim = 100,index_string="IVF1100,PQ100x4fsr,RFlat",nprobe=77,k=340) to build an initial graph of the data.
### Graph refinement

We then performed a round of nndescent on the initial graph using a modified pynndescent to update the neighbour index I by reversing the neighbours and looking up the neighbours' neighbours.
## Running

> python3 knng.py contest-data-release-10m.bin

## Reference 

We use two open source python libraries: faiss and pynndescent, whose open source licenses are as follows:

1.[faiss](https://github.com/facebookresearch/faiss/blob/main/LICENSE)

2.[pynndescent](https://github.com/lmcinnes/pynndescent/blob/master/LICENSE)
