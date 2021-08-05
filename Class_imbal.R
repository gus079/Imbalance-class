library(tidymodels)
library(tidyverse)
library(skimr)
library(GGally)
theme_set(theme_bw())


## Load dataset

train_raw <- read_csv('train.csv', guess_max = 1e5) %>% 
  mutate(damaged= ifelse(damaged== 0, "no damaged", "damaged"))
test_raw <- read_csv('test.csv', guess_max = 1e5)


skim(train_raw)
skim(test_raw)

train_raw %>% 
  select(damaged, incident_year, height, speed, distance) %>% 
  ggpairs(columns = 2:5, aes(color = damaged, alpha = .5))

train_raw %>% 
  select(damaged, precipitation, visibility, engine_type, flight_impact, flight_phase, species_quantity) %>% 
  pivot_longer(precipitation:species_quantity) %>% 
  ggplot(aes(y= value, fill= damaged)) + 
  geom_bar(position = 'fill') +
  facet_wrap(vars(name), scales = 'free') + 
  labs(x= NULL, y= NULL, fill= NULL)


bird_df <- train_raw %>% 
  select(damaged, flight_impact, precipitation, visibility, flight_phase, engines, incident_year,
         incident_month, species_id, engine_type, aircraft_model, species_quantity,
         height, speed)


set.seed(123)

flight_folds <- vfold_cv(train_raw, v = 5, strata = damaged)

flight_metrics <- metric_set(mn_log_loss, accuracy, sensitivity, specificity)


bird_rec <- recipe(damaged ~ ., data = bird_df) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_unknown(all_nominal_predictors()) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_zv(all_predictors())
  
bird_df %>% count(damaged)

library(baguette)

bag_spec <- bag_tree(min_n = 10) %>%
  set_engine('rpart', times = 25) %>% 
  set_mode('classification')

imb_wf <- workflow() %>% 
  add_recipe(bird_rec) %>% 
  add_model(bag_spec)

fit(imb_wf, data= bird_df)


##Resample and comparing

doParallel::registerDoParallel()

set.seed(123)
imb_rs <- fit_resamples(
  imb_wf,
  resamples = flight_folds,
  metrics = flight_metrics)

collect_metrics(imb_rs)

library(themis)

bal_rec <- bird_rec %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(damaged)

bal_wf <- workflow() %>% 
  add_recipe(bal_rec) %>% 
  add_model(bag_spec)

set.seed(123)
bal_rs <- fit_resamples(
  bal_wf,
  resamples = flight_folds,
  metrics = flight_metrics)

collect_metrics(bal_rs)


























