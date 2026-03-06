# Improving Model Performance




## Big Idea: Many Models Beat One Model

Many machine learning algorithms are unstable. Small changes in the training data can lead to very different models. Decision trees are a classic example: change a few rows and the first split can change completely.

Ensemble methods address this problem by building many models and combining them.

Instead of trusting one model, we let a crowd of models vote.

Three common ensemble strategies are:

* **Bagging (Bootstrap Aggregating)** – train many models on bootstrap samples of the data and average their predictions.
* **Boosting** – train models sequentially so that each new model focuses on correcting the errors of the previous one.
* **Stacking** – combine predictions from multiple model types using a meta-model.

In this module we focus mostly on bagging, which leads directly to one of the most widely used algorithms in machine learning: random forests.

### Why Bagging Works Well for Trees

Decision trees have high variance. If we train the same tree algorithm on slightly different samples of the data we often get very different trees.

That variability hurts prediction performance.

Bagging fixes this by:

1. Drawing many bootstrap samples from the training data.
2. Fitting a tree to each sample.
3. Combining predictions across all trees (majority vote for classification).

Each individual tree may be noisy, but the average of many trees is usually more stable and more accurate.

This is the same idea behind the "wisdom of crowds."

## Packages

``` r
library(tidyverse) # for general use
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ dplyr     1.1.4     ✔ readr     2.1.6
## ✔ forcats   1.0.1     ✔ stringr   1.6.0
## ✔ ggplot2   4.0.1     ✔ tibble    3.3.1
## ✔ lubridate 1.9.4     ✔ tidyr     1.3.2
## ✔ purrr     1.2.1     
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(rpart) # for decision trees
library(visNetwork) # for plotting DT
library(ipred) # for bagging
library(randomForest) # for rf
```

```
## randomForest 4.7-1.2
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

``` r
library(caret) # and to tune the rf
```

```
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:purrr':
## 
##     lift
```

``` r
library(gbm) # for boosting
```

```
## Loaded gbm 2.2.3
## This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3
```

## Reading
Chapter 11: Improving Model Performance in Machine Learning with R: Expert techniques for predictive modeling, 3rd Edition. Link on Canvas.

## Decision Tree to Bagging to Tuning
I'm going to show you four different approaches to the same classification problem. The first is a straight decision tree with `rpart`. The second is a "forest" of bagged trees. The third is a full "random forest." And finally, we can tune the random forest approach. At each step we will look at performance on withheld testing data. 

### The Modeling Ladder

We will build models of increasing complexity:

| Method | Idea |
|------|------|
| Decision Tree (`rpart`) | One tree fit to the data |
| Bagging | Many trees trained on bootstrap samples |
| Random Forest | Bagging + random feature selection |
| Tuned Random Forest | Random forest with hyperparameters optimized |

Each step generally reduces variance and improves prediction accuracy.

### Fish Your Wish
Let's go back to the fish data we used in the initial kNN module. Go back and take a look if you don't remember. I'll read the data in, filter it a bit, drop the `NA` values, and make the common name a factor.


``` r
fish <- read_csv("data/fishcatch.csv")
```

```
## Rows: 159 Columns: 11
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (3): std_name, common_name, sex
## dbl (8): weight_g, length_nose2tail_base_cm, length_nose2tail_notch_cm, leng...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
fishFiltered <- fish %>% select(-std_name, -sex) %>%
  drop_na() %>%
  mutate(common_name = factor(common_name))
```

### Train and Test

Because we are working with a reasonably large dataset (for our purposes), we'll split the data for training and testing.

Instead of randomly sampling rows, we use `createDataPartition()` from `caret`, which performs a stratified split. This keeps the proportion of each fish species roughly the same in both the training and testing sets.


``` r
set.seed(3) # for reproducibility 
rows2test <- createDataPartition(fishFiltered$common_name, 
                                 p = 1/2, 
                                 list = FALSE)[,1]

testingFish  <- fishFiltered[rows2test, ]
trainingFish <- fishFiltered[-rows2test, ]
```

### Straight Decision Tree
Here is the simplest version of this classification task. I'll use `rpart` here. This is conceptually the same as using `C50` which we used in the decision tree. I want you to be aware that `rpart` is used commonly for classification as well as regression. Some of the details about how the splitting criteria are setup is different from `C50` but it's very similar. And `rpart` plays nicely with the `visTree` function which is nice. 


``` r
rpartModel <- rpart(common_name~.,
                    data=trainingFish)
visTree(rpartModel)
```

```{=html}
<div id="htmlwidget-8746836e2be25030c0ef" style="width:100%;height:600px;" class="visNetwork html-widget"></div>
<script type="application/json" data-for="htmlwidget-8746836e2be25030c0ef">{"x":{"nodes":{"id":[1,2,4,5,3,6,7],"label":["height2length_pct","weight_g","bream","silver_bream","height2length_pct","perch","pike"],"level":[1,2,3,3,2,3,3],"color":["#F1B8C2","#B4C7ED","#7D91B6","#B98099","#F1B8C2","#9F87B5","#AC83AE"],"value":[78,22,15,7,56,41,15],"shape":["dot","dot","square","square","dot","square","square"],"title":["<div style=\"text-align:center;\">N : <b>100%<\/b> (78)<br>Complexity : <b>0.34<\/b><br>bream : <b>21.8%<\/b> (17)<br>ide : <b>3.8%<\/b> (3)<br>perch : <b>35.9%<\/b> (28)<br>pike : <b>10.3%<\/b> (8)<br>roach : <b>12.8%<\/b> (10)<br>silver_bream : <b>6.4%<\/b> (5)<br>smelt : <b>9%<\/b> (7)<\/div>","<div style=\"text-align:center;\">N : <b>28.2%<\/b> (22)<br>Complexity : <b>0.06<\/b><br>bream : <b>77.3%<\/b> (17)<br>ide : <b>0%<\/b> (0)<br>perch : <b>0%<\/b> (0)<br>pike : <b>0%<\/b> (0)<br>roach : <b>0%<\/b> (0)<br>silver_bream : <b>22.7%<\/b> (5)<br>smelt : <b>0%<\/b> (0)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b> height2length_pct <\/b> >= 33.45<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>","<div style=\"text-align:center;\">N : <b>19.2%<\/b> (15)<br>Complexity : <b>0.01<\/b><br>bream : <b>100%<\/b> (15)<br>ide : <b>0%<\/b> (0)<br>perch : <b>0%<\/b> (0)<br>pike : <b>0%<\/b> (0)<br>roach : <b>0%<\/b> (0)<br>silver_bream : <b>0%<\/b> (0)<br>smelt : <b>0%<\/b> (0)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b> height2length_pct <\/b> >= 33.45<br><b> weight_g <\/b> >= 395<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>","<div style=\"text-align:center;\">N : <b>9%<\/b> (7)<br>Complexity : <b>0.01<\/b><br>bream : <b>28.6%<\/b> (2)<br>ide : <b>0%<\/b> (0)<br>perch : <b>0%<\/b> (0)<br>pike : <b>0%<\/b> (0)<br>roach : <b>0%<\/b> (0)<br>silver_bream : <b>71.4%<\/b> (5)<br>smelt : <b>0%<\/b> (0)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b> height2length_pct <\/b> >= 33.45<br><b> weight_g <\/b> < 395<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>","<div style=\"text-align:center;\">N : <b>71.8%<\/b> (56)<br>Complexity : <b>0.16<\/b><br>bream : <b>0%<\/b> (0)<br>ide : <b>5.4%<\/b> (3)<br>perch : <b>50%<\/b> (28)<br>pike : <b>14.3%<\/b> (8)<br>roach : <b>17.9%<\/b> (10)<br>silver_bream : <b>0%<\/b> (0)<br>smelt : <b>12.5%<\/b> (7)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b> height2length_pct <\/b> < 33.45<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>","<div style=\"text-align:center;\">N : <b>52.6%<\/b> (41)<br>Complexity : <b>0<\/b><br>bream : <b>0%<\/b> (0)<br>ide : <b>7.3%<\/b> (3)<br>perch : <b>68.3%<\/b> (28)<br>pike : <b>0%<\/b> (0)<br>roach : <b>24.4%<\/b> (10)<br>silver_bream : <b>0%<\/b> (0)<br>smelt : <b>0%<\/b> (0)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b>  <\/b> 21.45 <= <b>height2length_pct<\/b> < 33.45<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>","<div style=\"text-align:center;\">N : <b>19.2%<\/b> (15)<br>Complexity : <b>0.01<\/b><br>bream : <b>0%<\/b> (0)<br>ide : <b>0%<\/b> (0)<br>perch : <b>0%<\/b> (0)<br>pike : <b>53.3%<\/b> (8)<br>roach : <b>0%<\/b> (0)<br>silver_bream : <b>0%<\/b> (0)<br>smelt : <b>46.7%<\/b> (7)<hr class = \"rPartvisNetwork\">\n<div class =\"showOnMe2\"><div style=\"text-align:center;\"><U style=\"color:blue;\"  onmouseover=\"this.style.cursor='pointer';\" onmouseout=\"this.style.cursor='default';\">Rules<\/U><\/div>\n<div class=\"showMeRpartTTp2\" style=\"display:none;\">\n<b> height2length_pct <\/b> < 21.45<\/script><script type=\"text/javascript\">$(document).ready(function(){\n$(\".showOnMe2\").click(function(){\n$(\".showMeRpartTTp2\").toggle();\n$.sparkline_display_visible();\n});\n  });<\/script><\/div><\/div>\n\n<\/div>"],"fixed":[true,true,true,true,true,true,true],"colorClust":["#9F87B5","#7D91B6","#7D91B6","#B98099","#9F87B5","#9F87B5","#AC83AE"],"labelClust":["perch","bream","bream","silver_bream","perch","perch","pike"],"Leaf":[0,0,1,1,0,1,1],"font.size":[16,16,16,16,16,16,16],"scaling.min":[22.5,22.5,22.5,22.5,22.5,22.5,22.5],"scaling.max":[22.5,22.5,22.5,22.5,22.5,22.5,22.5]},"edges":{"id":["edge1","edge2","edge3","edge4","edge5","edge6"],"from":[1,2,2,1,3,3],"to":[2,4,5,3,6,7],"label":[">= 33.45",">= 395","< 395","< 33.45",">= 21.45","< 21.45"],"value":[22,15,7,56,41,15],"title":["<div style=\"text-align:center;\"><b>height2length_pct<\/b><\/div><div style=\"text-align:center;\">>=33.45<\/div>","<div style=\"text-align:center;\"><b>weight_g<\/b><\/div><div style=\"text-align:center;\">>=395<\/div>","<div style=\"text-align:center;\"><b>weight_g<\/b><\/div><div style=\"text-align:center;\"><395<\/div>","<div style=\"text-align:center;\"><b>height2length_pct<\/b><\/div><div style=\"text-align:center;\"><33.45<\/div>","<div style=\"text-align:center;\"><b>height2length_pct<\/b><\/div><div style=\"text-align:center;\">>=21.45<\/div>","<div style=\"text-align:center;\"><b>height2length_pct<\/b><\/div><div style=\"text-align:center;\"><21.45<\/div>"],"color":["#8181F7","#8181F7","#8181F7","#8181F7","#8181F7","#8181F7"],"font.size":[14,14,14,14,14,14],"font.align":["horizontal","horizontal","horizontal","horizontal","horizontal","horizontal"],"smooth.enabled":[true,true,true,true,true,true],"smooth.type":["cubicBezier","cubicBezier","cubicBezier","cubicBezier","cubicBezier","cubicBezier"],"smooth.roundness":[0.5,0.5,0.5,0.5,0.5,0.5]},"nodesToDataframe":true,"edgesToDataframe":true,"options":{"width":"100%","height":"100%","nodes":{"shape":"dot"},"manipulation":{"enabled":false},"layout":{"hierarchical":{"enabled":true,"direction":"UD"}},"interaction":{"dragNodes":false,"selectConnectedEdges":false,"tooltipDelay":500,"zoomSpeed":1},"edges":{"scaling":{"label":{"enabled":false}}}},"groups":null,"width":"100%","height":"600px","idselection":{"enabled":false,"style":"width: 150px; height: 26px","useLabels":true,"main":"Select by id"},"byselection":{"enabled":false,"style":"width: 150px; height: 26px","multiple":false,"hideColor":"rgba(200,200,200,0.5)","highlight":false},"main":{"text":"","style":"font-family:Georgia, Times New Roman, Times, serif;font-weight:bold;font-size:20px;text-align:center;"},"submain":{"text":"","style":"font-family:Georgia, Times New Roman, Times, serif;font-size:12px;text-align:center;"},"footer":{"text":"","style":"font-family:Georgia, Times New Roman, Times, serif;font-size:12px;text-align:center;"},"background":"rgba(0, 0, 0, 0)","highlight":{"enabled":true,"hoverNearest":false,"degree":{"from":50000,"to":0},"algorithm":"hierarchical","hideColor":"rgba(200,200,200,0.5)","labelOnly":true},"collapse":{"enabled":true,"fit":true,"resetHighlight":true,"clusterOptions":{"fixed":true,"physics":false},"keepCoord":true,"labelSuffix":"(cluster)"},"tooltipStay":300,"tooltipStyle":"position: fixed;visibility:hidden;padding: 5px;\n                      white-space: nowrap;\n                      font-family: cursive;font-size:12px;font-color:purple;background-color: #E6E6E6;\n                      border-radius: 15px;","OnceEvents":{"stabilized":"function() { \n        this.setOptions({layout:{hierarchical:false}, physics:{solver:'barnesHut', enabled:true, stabilization : false}, nodes : {physics : false, fixed : true}});\n    }"},"legend":{"width":0.1,"useGroups":false,"position":"left","ncol":1,"stepX":100,"stepY":100,"zoom":true,"nodes":{"label":["height2length_pct","weight_g","bream","ide","perch","pike","roach","silver_bream","smelt"],"color":["#F1B8C2","#B4C7ED","#7D91B6","#8F8CB7","#9F87B5","#AC83AE","#B481A5","#B98099","#B9828C"],"shape":["dot","dot","square","square","square","square","square","square","square"],"size":[22,22,22,22,22,22,22,22,22],"Leaf":[0,0,1,1,1,1,1,1,1],"font.size":[16,16,16,16,16,16,16,16,16],"id":[10000,10001,10002,10003,10004,10005,10006,10007,10008]},"nodesToDataframe":true},"tree":{"updateShape":true,"shapeVar":"dot","shapeY":"square","colorVar":{"variable":["height2length_pct","weight_g"],"color":["#F1B8C2","#B4C7ED"]},"colorY":{"colorY":{"modality":["bream","ide","perch","pike","roach","silver_bream","smelt"],"color":["#7D91B6","#8F8CB7","#9F87B5","#AC83AE","#B481A5","#B98099","#B9828C"]},"vardecidedClust":["perch","bream","bream","silver_bream","perch","perch","pike"]}},"export":{"type":"png","css":"float:right;-webkit-border-radius: 10;\n                  -moz-border-radius: 10;\n                  border-radius: 10px;\n                  font-family: Arial;\n                  color: #ffffff;\n                  font-size: 12px;\n                  background: #090a0a;\n                  padding: 4px 8px 4px 4px;\n                  text-decoration: none;","background":"#fff","name":"network.png","label":"Export as png"}},"evals":["OnceEvents.stabilized"],"jsHooks":[]}</script>
```

One of the things to beware of is that this is the last graphical tree you'll see as we move forward. Because we are going to on to forests of trees, there is no easy visualization with other methods. Instead we interpret via performance metrics like accuarcy and kappa.

And here is the confusion matrix on the testing data.


``` r
obs <- testingFish$common_name
pred <- predict(rpartModel,testingFish, type = "class")
rpartCM <- confusionMatrix(data = pred, reference = obs)
rpartCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           14   0     0    0     0            0     0
##   ide              0   0     0    0     0            0     0
##   perch            0   3    27    0    10            0     0
##   pike             0   0     1    9     0            0     7
##   roach            0   0     0    0     0            0     0
##   silver_bream     3   0     0    0     0            6     0
##   smelt            0   0     0    0     0            0     0
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.5872, 0.7974)
##     No Information Rate : 0.35            
##     P-Value [Acc > NIR] : 1.984e-10       
##                                           
##                   Kappa : 0.6029          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                0.8235     0.0000       0.9643      1.0000
## Specificity                1.0000     1.0000       0.7500      0.8873
## Pos Pred Value             1.0000        NaN       0.6750      0.5294
## Neg Pred Value             0.9545     0.9625       0.9750      1.0000
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.1750     0.0000       0.3375      0.1125
## Detection Prevalence       0.1750     0.0000       0.5000      0.2125
## Balanced Accuracy          0.9118     0.5000       0.8571      0.9437
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                 0.000              1.0000       0.0000
## Specificity                 1.000              0.9595       1.0000
## Pos Pred Value                NaN              0.6667          NaN
## Neg Pred Value              0.875              1.0000       0.9125
## Prevalence                  0.125              0.0750       0.0875
## Detection Rate              0.000              0.0750       0.0000
## Detection Prevalence        0.000              0.1125       0.0000
## Balanced Accuracy           0.500              0.9797       0.5000
```

### Bagging

Bagging stands for Bootstrap Aggregating.

The algorithm works like this:

1. Draw a bootstrap sample from the training data.
2. Train a decision tree on that sample.
3. Repeat this many times (often hundreds).
4. Combine predictions across trees.

Because each tree sees a slightly different dataset, the trees differ from each other. When we aggregate their predictions, random noise tends to cancel out.

The result is usually lower variance and better predictive performance than a single tree.

Now we will do a bootstrap aggregate of the same classification task with the `bagging` function. We will use 100 separate bootstrap replications (so 100 different trees).


``` r
baggedModel <- bagging(common_name~., 
                       data=trainingFish,
                       nbagg = 100,
                       coob = TRUE)
baggedModel
```

```
## 
## Bagging classification trees with 100 bootstrap replications 
## 
## Call: bagging.data.frame(formula = common_name ~ ., data = trainingFish, 
##     nbagg = 100, coob = TRUE)
## 
## Out-of-bag estimate of misclassification error:  0.2564
```

And the confusion matrix on the testing data. Same testing and training split, right?


``` r
pred <- predict(baggedModel,testingFish)
baggedCM <- confusionMatrix(data = pred, reference = obs)
baggedCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           17   0     0    0     0            0     0
##   ide              0   1     1    0     0            0     0
##   perch            0   2    23    0     3            0     0
##   pike             0   0     0    9     0            0     0
##   roach            0   0     3    0     7            0     0
##   silver_bream     0   0     0    0     0            6     0
##   smelt            0   0     1    0     0            0     7
## 
## Overall Statistics
##                                           
##                Accuracy : 0.875           
##                  95% CI : (0.7821, 0.9384)
##     No Information Rate : 0.35            
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8415          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                1.0000     0.3333       0.8214      1.0000
## Specificity                1.0000     0.9870       0.9038      1.0000
## Pos Pred Value             1.0000     0.5000       0.8214      1.0000
## Neg Pred Value             1.0000     0.9744       0.9038      1.0000
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.2125     0.0125       0.2875      0.1125
## Detection Prevalence       0.2125     0.0250       0.3500      0.1125
## Balanced Accuracy          1.0000     0.6602       0.8626      1.0000
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                0.7000               1.000       1.0000
## Specificity                0.9571               1.000       0.9863
## Pos Pred Value             0.7000               1.000       0.8750
## Neg Pred Value             0.9571               1.000       1.0000
## Prevalence                 0.1250               0.075       0.0875
## Detection Rate             0.0875               0.075       0.0875
## Detection Prevalence       0.1250               0.075       0.1000
## Balanced Accuracy          0.8286               1.000       0.9932
```
#### Reminder: Why is bootstrapping OK?
You might sometimes feel like bootstrapping is cheating. We are repeatedly sampling from the same dataset, so it can feel like we are just reusing the same information.

The important idea is that our dataset is not the population. It is only a sample from a much larger population of possible observations.

Bootstrapping treats the observed dataset as the best available approximation of that population. By sampling from it with replacement, we create many slightly different datasets that mimic the variability we might see if we repeatedly sampled from the real population.

Each bootstrap dataset contains:
- some observations multiple times
- some observations not at all

Those small differences are enough to produce slightly different models.

Bagging takes advantage of this. Instead of trusting one model trained on one dataset, we train many models on many bootstrap versions of the data and combine their predictions.

The goal is not to create new information. The goal is to expose how sensitive the model is to small changes in the data and stabilize the predictions by averaging across many models.

### Random Forest (aka randomForest, aka rF)
And now we will pull in the big algorithm. The `randomForest` approach is pretty ubiquitous in ML now. Make sure to read this section in the text and [watch the StatQuest guy explain it all](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&ab_channel=StatQuestwithJoshStarmer).

Random forests build on bagging but add an additional layer of randomness.

In bagging:
- every tree can consider all predictor variables when choosing a split.

In random forests:
- each split only considers a random subset of predictors.

Why do this?

If one predictor is very strong, bagged trees will tend to split on it first every time. This makes the trees highly correlated.

Random forests force trees to explore different predictors, making the trees less correlated, which improves the ensemble when we average them.

#### Bagging vs Random Forest? 
Both methods train many trees on bootstrap samples. In bagging, every split can consider all predictors. In a random forest, each split considers only a random subset of predictors (mtry). This reduces correlation among trees and usually improves prediction. Or a simple way I think about it is that  bagging uses different rows from the data and random forest uses different rows and different columns.

Let's grow a random forest of 100 trees.


``` r
rfModel <- randomForest(common_name~.,
                        ntree = 100,
                        data=trainingFish)
rfModel
```

```
## 
## Call:
##  randomForest(formula = common_name ~ ., data = trainingFish,      ntree = 100) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 32.05%
## Confusion matrix:
##              bream ide perch pike roach silver_bream smelt class.error
## bream           16   0     0    0     0            1     0  0.05882353
## ide              0   0     3    0     0            0     0  1.00000000
## perch            0   1    18    0     7            0     2  0.35714286
## pike             0   0     0    8     0            0     0  0.00000000
## roach            0   0     9    0     1            0     0  0.90000000
## silver_bream     1   0     0    0     0            4     0  0.20000000
## smelt            0   0     1    0     0            0     6  0.14285714
```

And the confusion matrix on the testing data.


``` r
pred <- predict(rfModel,testingFish)
rfCM <- confusionMatrix(data = pred, reference = obs)
rfCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           17   0     0    0     0            0     0
##   ide              0   1     0    0     0            0     0
##   perch            0   2    24    0     2            0     0
##   pike             0   0     0    9     0            0     0
##   roach            0   0     4    0     8            0     0
##   silver_bream     0   0     0    0     0            6     0
##   smelt            0   0     0    0     0            0     7
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9             
##                  95% CI : (0.8124, 0.9558)
##     No Information Rate : 0.35            
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.873           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                1.0000     0.3333       0.8571      1.0000
## Specificity                1.0000     1.0000       0.9231      1.0000
## Pos Pred Value             1.0000     1.0000       0.8571      1.0000
## Neg Pred Value             1.0000     0.9747       0.9231      1.0000
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.2125     0.0125       0.3000      0.1125
## Detection Prevalence       0.2125     0.0125       0.3500      0.1125
## Balanced Accuracy          1.0000     0.6667       0.8901      1.0000
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                0.8000               1.000       1.0000
## Specificity                0.9429               1.000       1.0000
## Pos Pred Value             0.6667               1.000       1.0000
## Neg Pred Value             0.9706               1.000       1.0000
## Prevalence                 0.1250               0.075       0.0875
## Detection Rate             0.1000               0.075       0.0875
## Detection Prevalence       0.1500               0.075       0.0875
## Balanced Accuracy          0.8714               1.000       1.0000
```

### A Tuned Random Forest
The main tuning parameter in a random forest is `mtry`, the number of predictors randomly considered at each split.

Small values increase tree diversity but may weaken individual trees. Larger values make trees more similar.

Cross-validation helps find a good balance.

So let's try to tune this model.


``` r
cvScheme <- trainControl(method = "repeatedcv",
                         number=5,repeats = 20)
rfTuningGrid <- data.frame(mtry = 2:6)

rfTuned <- train(common_name~.,
                 data=trainingFish,
                 method="rf",
                 trControl = cvScheme,
                 tuneGrid = rfTuningGrid)
rfTuned
```

```
## Random Forest 
## 
## 78 samples
##  8 predictor
##  7 classes: 'bream', 'ide', 'perch', 'pike', 'roach', 'silver_bream', 'smelt' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 20 times) 
## Summary of sample sizes: 64, 60, 62, 63, 63, 62, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.7317534  0.6487901
##   3     0.7386554  0.6596274
##   4     0.7428917  0.6650762
##   5     0.7485912  0.6722428
##   6     0.7552825  0.6816600
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 6.
```

And the confusion matrix on the testing data.


``` r
pred <- predict(rfTuned,testingFish)
rfTunedCM <- confusionMatrix(data = pred, reference = obs)
rfTunedCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           17   0     0    0     0            0     0
##   ide              0   1     0    0     0            0     0
##   perch            0   2    25    0     2            0     0
##   pike             0   0     0    9     0            0     0
##   roach            0   0     3    0     8            0     0
##   silver_bream     0   0     0    0     0            6     0
##   smelt            0   0     0    0     0            0     7
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9125         
##                  95% CI : (0.828, 0.9641)
##     No Information Rate : 0.35           
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.8884         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                1.0000     0.3333       0.8929      1.0000
## Specificity                1.0000     1.0000       0.9231      1.0000
## Pos Pred Value             1.0000     1.0000       0.8621      1.0000
## Neg Pred Value             1.0000     0.9747       0.9412      1.0000
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.2125     0.0125       0.3125      0.1125
## Detection Prevalence       0.2125     0.0125       0.3625      0.1125
## Balanced Accuracy          1.0000     0.6667       0.9080      1.0000
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                0.8000               1.000       1.0000
## Specificity                0.9571               1.000       1.0000
## Pos Pred Value             0.7273               1.000       1.0000
## Neg Pred Value             0.9710               1.000       1.0000
## Prevalence                 0.1250               0.075       0.0875
## Detection Rate             0.1000               0.075       0.0875
## Detection Prevalence       0.1375               0.075       0.0875
## Balanced Accuracy          0.8786               1.000       1.0000
```

### Skill
While there is lots and lots of information in each of the confusion matrices, here is what most folks think is the bottom line for classification skill.

We will focus on two summary metrics:

* **Accuracy** – proportion of correct predictions
* **Kappa** – improvement over random guessing

Higher values indicate better classification performance.


``` r
rbind(rpart=rpartCM$overall[1:2],
      bagging=baggedCM$overall[1:2],
      randomForest = rfCM$overall[1:2],
      tunedRandomForest = rfTunedCM$overall[1:2])
```

```
##                   Accuracy     Kappa
## rpart               0.7000 0.6028956
## bagging             0.8750 0.8415214
## randomForest        0.9000 0.8729655
## tunedRandomForest   0.9125 0.8884462
```


## Takeaway

Decision trees are easy to understand but often unstable.

Bagging stabilizes trees by averaging across many bootstrap samples.

Random forests go one step further by forcing trees to explore different predictors, producing a strong and reliable ensemble.

## But where is the tree?

If you trained a single decision tree earlier in the course, you could draw it.  
There was one model and it had a visible structure.

Bagging and random forests are different.

Instead of fitting one tree, these methods fit many trees. Often hundreds.

Each tree is trained on:
- a bootstrap sample of the rows, and
- (for random forests) a random subset of predictors when choosing splits.

We can look at variable importance with something like `varImp(rfTuned)` to get an idea of what the critical drivers are but forests are much more opaque than individual trees in terms of inference.

### How can they predict new data?

You might notice something odd: each tree only sees a bootstrap sample of the training data. Some observations are left out. So how can the model make predictions for new data?

The important thing to understand is that a decision tree does not memorize rows. It just learns rules about predictor values.

For example a tree might learn rules like:

```
if length < 25 → perch
if length ≥ 25 and height < 12 → roach
if length ≥ 25 and height ≥ 12 → bream
```

Those rules apply to any observation with those predictor values, whether or not that observation appeared in the bootstrap sample.

So when we make a prediction for a new data point:

1. The observation is sent down every tree in the ensemble.
2. Each tree produces a prediction.
3. The predictions are combined.

For classification:
- trees vote, and the majority class wins.

For regression:
- predictions are averaged.

### Which trees are used?

All of them.

If the model contains 500 trees, the new observation is run through all 500 trees, and their predictions are aggregated.

The individual trees are usually noisy and not very interesting on their own. The strength of bagging and random forests comes from the combined prediction of many trees.

## Your Work
Let's assume that you are hip to the general idea of how `rpart` is doing its classification. Remember it works like the C5.0 algorithm with some differences in how many samples are required for a split. But it's essentially the same as C5.0.

For each bullet below write a paragraph, or so, explaining to your peer reviewer:

* How `bagging` differs from the straight `rpart` approach,

* How `randomForest` differs from `bagging` and,

* How the `caret` approach using `train` differs from `randomForest`.

Next, try something nutty: Go find a model we have never used. Start by looking [here](https://topepo.github.io/caret/models-clustered-by-tag-similarity.html). Pick either a classification or a dual-use model (can be used for classification or regression) and run it on the fish data. 

For instance here is a "Model Averaged Neural Network" without paying any attention to the tuning or cross validation.


``` r
avNNetModel <- train(common_name~.,data=trainingFish,
                     method="avNNet",
                     preProcess = "range", # recall that NN need preprocessing!
                     trace = FALSE) # makes it quiet
```

```
## Warning: executing %dopar% sequentially: no parallel backend registered
```

``` r
avNNetModel
```

```
## Model Averaged Neural Network 
## 
## 78 samples
##  8 predictor
##  7 classes: 'bream', 'ide', 'perch', 'pike', 'roach', 'silver_bream', 'smelt' 
## 
## Pre-processing: re-scaling to [0, 1] (8) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 78, 78, 78, 78, 78, 78, ... 
## Resampling results across tuning parameters:
## 
##   size  decay  Accuracy   Kappa    
##   1     0e+00  0.5190170  0.3002396
##   1     1e-04  0.5791789  0.4176441
##   1     1e-01  0.5441244  0.3391690
##   3     0e+00  0.5294712  0.3540930
##   3     1e-04  0.6654848  0.5386148
##   3     1e-01  0.7180472  0.6040233
##   5     0e+00  0.4761047  0.2772846
##   5     1e-04  0.6817720  0.5561172
##   5     1e-01  0.7415262  0.6396381
## 
## Tuning parameter 'bag' was held constant at a value of FALSE
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were size = 5, decay = 0.1 and bag = FALSE.
```

``` r
pred <- predict(avNNetModel,testingFish)
avNNetCM <- confusionMatrix(data = pred, reference = obs)
avNNetCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           17   0     0    0     0            6     0
##   ide              0   0     0    0     0            0     0
##   perch            0   3    28    0    10            0     0
##   pike             0   0     0    9     0            0     0
##   roach            0   0     0    0     0            0     0
##   silver_bream     0   0     0    0     0            0     0
##   smelt            0   0     0    0     0            0     7
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7625          
##                  95% CI : (0.6542, 0.8505)
##     No Information Rate : 0.35            
##     P-Value [Acc > NIR] : 5.955e-14       
##                                           
##                   Kappa : 0.6787          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                1.0000     0.0000       1.0000      1.0000
## Specificity                0.9048     1.0000       0.7500      1.0000
## Pos Pred Value             0.7391        NaN       0.6829      1.0000
## Neg Pred Value             1.0000     0.9625       1.0000      1.0000
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.2125     0.0000       0.3500      0.1125
## Detection Prevalence       0.2875     0.0000       0.5125      0.1125
## Balanced Accuracy          0.9524     0.5000       0.8750      1.0000
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                 0.000               0.000       1.0000
## Specificity                 1.000               1.000       1.0000
## Pos Pred Value                NaN                 NaN       1.0000
## Neg Pred Value              0.875               0.925       1.0000
## Prevalence                  0.125               0.075       0.0875
## Detection Rate              0.000               0.000       0.0875
## Detection Prevalence        0.000               0.000       0.0875
## Balanced Accuracy           0.500               0.500       1.0000
```

Skill is meh.

What if we wanted to tune it? We can see what parameters are available via `modelLookup("avNNet")`


``` r
modelLookup("avNNet")
```

```
##    model parameter         label forReg forClass probModel
## 1 avNNet      size #Hidden Units   TRUE     TRUE      TRUE
## 2 avNNet     decay  Weight Decay   TRUE     TRUE      TRUE
## 3 avNNet       bag       Bagging   TRUE     TRUE      TRUE
```

This tells us that there are three tweaks we can make to try to improve performance: `size`, `decay`, and `bag`. What are reasonable values for those? I have no idea! But looking at the help for `avNNet` tells us that it calls `nnet`. By looking at those I was able to make an initial stab at figuring it out. I'm still not sure if these are in the right range for this algorithm frankly but it's a start. Oh and I'll cross validate it too.

Warning. This will take awhile to run.


``` r
avNNetTuningGrid <- expand.grid(bag = c(TRUE,FALSE),
                                decay = seq(0,0.01,by=0.005),
                                size = c(1,3,5,10,20))

avNNetModelTuned <- train(common_name~.,data=trainingFish,
                     method="avNNet",
                     preProcess = "range",
                     trControl = cvScheme,
                     tuneGrid = avNNetTuningGrid,
                     trace = FALSE)
avNNetModelTuned

pred <- predict(avNNetModelTuned,testingFish)
avNNetTunedCM <- confusionMatrix(data = pred, reference = obs)

avNNetTunedCM
```

This worked well on the testing data.

What monstrous algorithm did you unleash on the data? A kNN variant? A SVM? Neural network? Multivariate Adaptive Regression Spline? Model Tree? Radial Basis Function? Self-Organizing Map? Did it work well on the testing data? Do you have any idea what it does? How did you tune it?

## Postscript: Boosting and Stacking

### Boosting
I've given you a lot above! But I'd be remiss if I didn't crack the door open for boosting which is a very common technique.

Bagging and random forests build many independent trees and combine their predictions.

Boosting takes a different approach.

Instead of training trees independently, boosting builds trees sequentially. Each new tree focuses on correcting the errors made by the previous trees.

The idea is simple:

1. Fit a small tree to the data.
2. Identify where that tree makes mistakes.
3. Fit another tree that focuses on those mistakes.
4. Repeat many times.

The final prediction combines the contribution of all trees.

#### A helpful way to think about boosting

Bagging and random forests rely on the wisdom of crowds. Many independent trees vote.

Boosting is more like an iterative improvement process. Each tree tries to fix the remaining errors.

#### Weak learners

Boosting typically uses very small trees, sometimes called *weak learners*. These trees might be:

- shallow
- simple
- not very accurate on their own

But when hundreds of them are added together, the combined model can become very powerful.

#### How predictions work

Even though trees are built sequentially, predictions still use all trees in the model.

For regression:
- predictions are added together across trees.

For classification:
- trees contribute weighted votes that are combined into a final prediction.

#### The key difference from bagging and random forests

Bagging and random forests:
- trees are built independently
- trees reduce variance

Boosting:
- trees are built sequentially
- trees reduce bias by correcting errors

Because of this, boosting models can often outperform random forests, but they are usually more sensitive to tuning. 

Here is an untuned boosting attempt.


``` r
gbmCaret <- train(common_name~.,
                  data=trainingFish,
                  method="gbm",
                  trControl = cvScheme,
                  verbose = FALSE) # makes it quiet
gbmCaret
```

```
## Stochastic Gradient Boosting 
## 
## 78 samples
##  8 predictor
##  7 classes: 'bream', 'ide', 'perch', 'pike', 'roach', 'silver_bream', 'smelt' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 20 times) 
## Summary of sample sizes: 61, 62, 62, 65, 62, 63, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7664745  0.6965524
##   1                  100      0.7532854  0.6801043
##   1                  150      0.7421058  0.6658439
##   2                   50      0.7692778  0.7006079
##   2                  100      0.7548527  0.6826783
##   2                  150      0.7394383  0.6637660
##   3                   50      0.7708838  0.7025738
##   3                  100      0.7517874  0.6782875
##   3                  150      0.7411605  0.6655430
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 50, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

``` r
pred <- predict(gbmCaret,testingFish)
gbmCaretCM <- confusionMatrix(data = pred, reference = obs)
gbmCaretCM
```

```
## Confusion Matrix and Statistics
## 
##               Reference
## Prediction     bream ide perch pike roach silver_bream smelt
##   bream           17   0     0    0     0            0     0
##   ide              0   0     0    0     0            0     0
##   perch            0   3    23    1     3            0     0
##   pike             0   0     0    8     0            0     0
##   roach            0   0     4    0     7            0     0
##   silver_bream     0   0     0    0     0            6     0
##   smelt            0   0     1    0     0            0     7
## 
## Overall Statistics
##                                         
##                Accuracy : 0.85          
##                  95% CI : (0.7526, 0.92)
##     No Information Rate : 0.35          
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.8079        
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: bream Class: ide Class: perch Class: pike
## Sensitivity                1.0000     0.0000       0.8214      0.8889
## Specificity                1.0000     1.0000       0.8654      1.0000
## Pos Pred Value             1.0000        NaN       0.7667      1.0000
## Neg Pred Value             1.0000     0.9625       0.9000      0.9861
## Prevalence                 0.2125     0.0375       0.3500      0.1125
## Detection Rate             0.2125     0.0000       0.2875      0.1000
## Detection Prevalence       0.2125     0.0000       0.3750      0.1000
## Balanced Accuracy          1.0000     0.5000       0.8434      0.9444
##                      Class: roach Class: silver_bream Class: smelt
## Sensitivity                0.7000               1.000       1.0000
## Specificity                0.9429               1.000       0.9863
## Pos Pred Value             0.6364               1.000       0.8750
## Neg Pred Value             0.9565               1.000       1.0000
## Prevalence                 0.1250               0.075       0.0875
## Detection Rate             0.0875               0.075       0.0875
## Detection Prevalence       0.1375               0.075       0.1000
## Balanced Accuracy          0.8214               1.000       0.9932
```

To tune this you can look fire up `modelLookup("gbm")` and then build a tuning grid. Note that this code will take awhile to run.


``` r
gbmTuningGrid <- expand.grid(
  n.trees = c(300, 600, 1000),
  interaction.depth = c(1, 2, 3),
  shrinkage = c(0.025, 0.05, 0.1),
  n.minobsinnode = c(5, 10)
)

gbmCaretTuned <- train(common_name~.,
                  data=trainingFish,
                  method="gbm",
                  trControl = cvScheme, 
                  tuneGrid = gbmTuningGrid,
                  verbose = FALSE)

gbmCaretTuned

pred <- predict(gbmCaretTuned,testingFish)
gbmCaretTunedCM <- confusionMatrix(data = pred, reference = obs)
gbmCaretTunedCM
```

## Stacking (briefly)

Bagging, random forests, and boosting combine many versions of the same kind of model (trees) in different ways.

Stacking is different: it combines different model types.

The idea:

1. Train several base models (for example: a random forest, a boosted tree model, and a k-nearest neighbors model).
2. Use cross-validation to generate out-of-sample predictions from each base model.
3. Train a meta-model that learns how to combine those predictions into a final prediction.

So the meta-model is not learning from the original predictors directly. It is learning from the predictions of other models.

Two important cautions:

- Stacking only makes sense if the base models make different kinds of mistakes. If they all fail on the same observations, stacking won’t help.
- You have to generate meta-model training data using out-of-sample predictions (typically from cross-validation), otherwise stacking can overfit badly.

We are not doing stacking in this module, but it is a common approach in machine learning competitions and in applied work when you want to squeeze out a bit more predictive performance.
