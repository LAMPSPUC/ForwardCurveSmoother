# ForwardCurveSmoother

This module contains the semi-parametric structural model for electricity forward curves develope by Lamps team, Marina Dietze and Iago Chávarry, under the coordination of Alexandre Street, Davi Valladão and Ana Carolina Freire. The methodology behind the package was developed and tested in collaboration with and Stein-Erik Fleten (NTNU). The development of the package was partially supported by CAPES, FAPERJ, CNPq and P&D ANEEL PD-07625-0219/2019.

To obtain the framework results, the user must execute the code available in the file 'run.jl', where the details of the input and output are presented. In the folder 'Data', the templates of the two necessary CSV files (contracts.csv and spot.csv) are accesible, while the folder 'Results' gather the exported outputs (reconstructed_prices.csv, elementary_prices.csv and elementary_errors.csv).
