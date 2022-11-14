# Laplacian Option Discovery
An implementation of [A Laplacian Framework for Option Discovery in Reinforcement Learning](http://proceedings.mlr.press/v70/machado17a/machado17a.pdf)

UNDER PROGRESS

Majority of my contributions thus far are related to visualising the eigenvectors of the graph Laplacian that Mahadevan 2005 dubbed as "Proto Value Functions" or PVFs, that encode information on the geometry of an environment.


## Environments

I have added functionality for any environment to be read in as a `.txt` file (see `environments`).

The following environments have been implemented thus far:
* Grid: 1 Square room from Machado et al. 2017 (see `environments/gridenv.txt`)
* Room: 4 room environment from Mahadevan 2005 & Machado et al. 2017 (see `environments/room.txt`)
* I Maze: I shaped environment from Machado et al. 2017 (see `environments/imaze.txt`)
* Rect: 1 Rectangle room from Stachenfeld et al. 2014 (see `environments/rect.txt`)
* Single Barrier: 1 thin partial barrier in a room from Stachenfeld et al. 2014 (see `environments/singlebarrier.txt`)
* Hairpin Maze: Multiple thin alternating partial barriers in a room from Stachenfeld et al. 2014 (see `environments/hairpin.txt`)
* HallEnv: 1 pixel wide path to help me understand the eigevectors as discrete standing waves (see `environments/hall.txt`)

