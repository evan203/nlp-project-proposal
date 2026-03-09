librar
library("tidyverse")
set.seed(123)

alpaca <- read_csv("./code/data-exploration/data/alpaca.csv",
                   show_col_types = FALSE)
beaver_tails <- read_csv("./code/data-exploration/data/beaver_tails.csv",
                         show_col_types = FALSE)

alpaca |> slice_sample(n = 3)
beaver_tails |> slice_sample(n = 3)

alpaca <- alpaca |> mutate(input = nchar(instruction) +
                             replace_na(nchar(input), 0),
                           output = nchar(output))
beaver_tails <- beaver_tails |> mutate(input = nchar(prompt),
                                       output = nchar(response))

alpaca_len_longer <- alpaca |> pivot_longer(cols = c(input, output),
                                            names_to = "segment")

beaver_tails_len_longer <- beaver_tails |> pivot_longer(cols = c(input, output),
                                                        names_to = "segment")

default_theme = theme(axis.title = element_text(size = 14),
                      axis.text = element_text(size = 12),
                      plot.title = element_text(size = 16),
                      strip.text = element_text(size = 14))

ggplot(data = alpaca_len_longer, aes(x = value)) +
  default_theme +
  labs(title = "Alpaca Input/Output Length Distribution",
       y = "Samples",
       x = "Character Length") +
  facet_wrap(~segment, nrow = 2) +
  geom_histogram() +
  scale_x_log10()

ggsave(filename = "./code/data-exploration/plots/alpaca_length.png",
       width = 10, height = 6, units = "in")

ggplot(data = beaver_tails_len_longer, aes(x = value)) +
  default_theme +
  labs(title = "BeaverTails Input/Output Length Distribution",
       y = "Samples",
       x = "Character Length") +
  facet_wrap(~segment, nrow = 2) +
  geom_histogram() +
  scale_x_log10()

ggsave(filename = "./code/data-exploration/plots/beaver_tails_length.png",
       width = 10, height = 6, units = "in")

beaver_tails_category_cols <- colnames(beaver_tails)[5:18]

beaver_tails_categories <- pivot_longer(beaver_tails,
                                        cols = all_of(beaver_tails_category_cols), # nolint: line_length_linter.
                                        names_to = "category") |>
  filter(value == TRUE)

beaver_tails_category_freq <- beaver_tails_categories |>
  group_by(category) |>
  summarize(cat_freq = n())

ggplot(data = beaver_tails_category_freq, aes(x = category, y = cat_freq)) +
  default_theme +
  labs(title = "Frequency of Harm Categories in BeaverTails",
       x = "Category",
       y = "Frequency") +
  geom_col() +
  scale_x_discrete(guide = guide_axis(angle = 45))

ggsave(filename = "./code/data-exploration/plots/beaver_tails_categories.png",
       width = 10, height = 8, units = "in")

beaver_tails_co_occurrence <- beaver_tails_categories |> full_join(beaver_tails_categories, # nolint: line_length_linter.
                                                                   by = "...1")

beaver_tails_co_occurrence_ct <- beaver_tails_co_occurrence |>
  group_by(category.x, category.y) |>
  summarize(co_occurrrence = n()) |>
  filter(category.x < category.y) |>
  arrange(desc(co_occurrrence)) |>
  head(10)

beaver_tails_co_occurrence_ct |>
  write_csv("./code/data-exploration/plots/beaver_tails_co_occurrence.csv")