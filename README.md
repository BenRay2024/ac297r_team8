# Optimizing Crowd Simulation Models for Disaster Preparedness (APCOMP 297R Codebase)
#### Ben Ray, Calliste Skouras, Cyrus Asgari, Rudra Barua

This repository contains the implementation and datasets for a pipeline to learn pedestrian dynamics solely from video data. It includes various baseline models, a CSRNet model for crowd counting, and a custom implementation of the Social Force Model adapted for specific scenarios.

## Repository Structure

- `pipeline`: Contains the main notebooks for the full pipeline described in the paper (designed to be used in a Google Colab environment).
    - `videoanalysis.ipynb`: Notebook for extracting crowd density maps from video data.
    - `optimization_pipeline.ipynb`: Notebook for optimizing the parameters of the Social Force Model using CMA-ES and visualizing the results.
- `baselines`: Contains notebooks for baseline models.
    - `LSTM_baseline.ipynb`
    - `random_baseline.ipynb`
- `csrnet/weights.pth`: Contains pre-trained weights for the CSRNet model.
- `data`: Contains the datasets used in the projects.
    - `GC_Dataset_ped1-12685_time1000-1060_interp9_xrange5-25_yrange15-35.npy`: Labelled trajectories from Grand Central Station.
    - `earthcam_ny_video.mp4`: Unlablled video from New York's Times Square bomb scare.
- `pysocialforce`: Python module implementing the Social Force Model adapted from [this GitHub repo](git@github.com:yuxiang-gao/PySocialForce.git). Key files include:
    - `config/default.toml`: Default configuration for the simulations.
    - `forces.py`: Implements various forces acting on pedestrians.
    - `scene.py`: Handles the simulation environment.
    - `simulator.py`: Core simulation functionality.
    - `utils/plot.py`: Plotting utilities.
- `LICENSE`: Contains the license information for the project.
- `README.md`: This file, providing an overview of the repository.

## Getting Started

To get started with this repository with the least friction, we recommend loading this entire repository into Google Drive, and using the `.ipynb` notebooks in a standard Colab environment.

You can then explore the notebooks and modules as follows:

1. Navigate to the `baselines` directory to run the baseline models.
2. Navigate to the `pipeline` directory to run the full pipeline outlined in our paper.
3. The `pysocialforce` directory contains the custom Social Force Model that can be integrated into your simulation environments.

## Contributing

Contributions to this project are welcome! Please reach out with any questions or suggestions to any of: 
- benray@college.harvard.edu
- cskouras@college.harvard.edu
- cyrusasgari@college.harvard.edu
- rudrabarua@college.harvard.edu

## License

This project is licensed under the [LICENSE](./LICENSE) found in this repository.