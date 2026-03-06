librar
library("tidyverse")
set.seed(123)

alpaca <- read_csv("./code/data-exploration/data/alpaca.csv",
                   show_col_types = FALSE)
beaver_tails <- read_csv("./code/data-exploration/data/beaver_tails.csv",
                         show_col_types = FALSE)

alpaca |> slice_sample(n = 3)
beaver_tails |> slice_sample(n = 3)

alpaca <- alpaca |> mutate(input_len = nchar(instruction) +
                             replace_na(nchar(input), 0),
                           output_len = nchar(output))
beaver_tails <- beaver_tails |> mutate(input_len = nchar(prompt),
                                       output_len = nchar(response))

alpaca_len_longer <- alpaca |> pivot_longer(cols = c(input_len, output_len),
                                            names_to = "segment")


ggplot(data = alpaca_len_longer, aes(x = value)) +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 16),
        strip.text = element_text(size = 14)) +
  labs(title = "Alpaca Input/Output Length Distribution") +
  facet_wrap(~segment, nrow = 2) +
  geom_histogram() +
  scale_x_log10()

ggsave(filename = "./code/data-exploration/plots/alpaca_length.png",
       width = 10, height = 6, units = "in")