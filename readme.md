## Running environment 
_____________________________________
Our program is written entirely in python and we write the dependency condition and library information of the program in requirements.txt.If you want to run the code successfully, please install these dependencies first.

Our program also uses pynndescent, but you do not need to download it. This is because we have made so many changes to the pyndescent source code that we have packaged it separately, in a folder named pynndescent_opt, where you can see our modified pyndescent code.
## Program introduction
_____________________________________
Our program consists of two main parts,initial graph construction and graph refinement.
### Initial graph construction
_____________________________________
After reading the content-data-release-10m.bin, we used faiss.index_factory(dim = 100,index_string="IVF1100,PQ100x4fsr,RFlat",nprobe=77,k=340) to build an initial graph of the data.
### Graph refinement
_____________________________________
We then performed a round of nndescent on the initial graph using a modified pynndescent to update the neighbour index I by reversing the neighbours and looking up the neighbours' neighbours.
## Running
_____________________________________
> python3 knng.py contest-data-release-10m.bin

## Reference 
_____________________________________
We use two open source python libraries: faiss and pynndescent, whose open source licenses are as follows:

1.[faiss] :https://github.com/facebookresearch/faiss/blob/main/LICENSE

2.[pynndescent]:https://github.com/lmcinnes/pynndescent/blob/master/LICENSE