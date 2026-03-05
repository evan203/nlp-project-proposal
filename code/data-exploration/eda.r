install.packages("arrow")
install.packages("tidyverse")
library("arrow")
library("tidyverse")

alpaca <- read_parquet("./code/data-exploration/data/alpaca.parquet")
beaver_tails <- read_parquet("./code/data-exploration/data/beaver_tails.parquet")

beaver_tails