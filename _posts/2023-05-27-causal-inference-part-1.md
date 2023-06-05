---
layout: post
cover: 'assets/images/covers/nebula-cover.jpeg'
navigation: True
title: Introduction to Causal Inference - Part 1
date: 2023-05-27 6:00:00
tags: [Causal Inference]
subclass: 'post tag-causalinference'
logo: 'assets/images/logo.png'
author: Austin
categories:
---

I've been thinking about this corner of the internet a lot lately. I just finished my master's degree in statistics at the University of Utah and to say I am relieved would be a huge understatement. I wanted to continue building this page while I was still in school, but my classes and master's project proved to be incredibly demanding on their own. On top of working full time, I just couldn't make technical blogging a priority. Now that I've finally graduated, I'm excited to share a lot of what I've learned (and am learning!) with the folks that stumble on my page.

- [Introduction: Housekeeping and Intentions](#introduction-housekeeping-and-intentions)
- [Setting the Stage for Causal Inference: Randomized Experiments](#setting-the-stage-for-causal-inference-randomized-experiments)
  - [Covariate Balancing in Randomized Experiments](#covariate-balancing-in-randomized-experiments)
  - [Fisher's Sharp Hypothesis of No Effect](#fishers-sharp-hypothesis-of-no-effect)
- [Stepping Toward Observational Studies](#stepping-toward-observational-studies)
  - [Simpson's Paradox](#simpsons-paradox)
- [What's Coming Next](#whats-coming-next)

## Introduction: Housekeeping and Intentions

During the last year, I've dived into learning about a fascinating, emergent branch of statistics called **causal inference**. It's exactly what it sounds like; in statistical inference, we are concerned with characterizing an association between variables, but you will often hear the old adage the correlation does not imply causation. Of course, this adage is (often annoyingly) true. Indeed, it would be unwise to believe that [per capita consumption of cheese in a population causes more people to die by becoming tangled in their bedsheets](https://www.tylervigen.com/spurious-correlations).

**Causal inference** is concerned with identifying causal relationships between variables. As you can imagine, characterizing this type of relationship is far more difficult and less mechanical than standard statistical inference techniques. There are a few competing schools of thought in this area as well, but the methods that I'll discuss in this series of posts about causal inference are derived from the research of [Dr. Paul Rosenbaum](https://statistics.wharton.upenn.edu/profile/rosenbap/), a statistician at the University of Pennsylvania.

Causal inference is a huge and complex topic, and so I want my treatment of it here to give it the respect it deserves. That's why this will be the first in a series of posts where I design a basic observational study from the ground up. These posts will most likely cover the following topics, roughly in this order:

- Defining Randomized Experiments
- Defining Observational Studies
- Matched Pair Designs, the Naïve Model, and Propensity Scores
- Pair Matching Algorithms
- Analyzing Results via Wilcoxon's Signed Rank Test
- Sensitivity Analysis and the Design Sensitivity Metric
- Applying Causal Methods to Build an Observational Study

Luckily, most of the basic concepts behind causal inference are relatively intuitive, and so the big ideas aren't extremely difficult to grapple with. However, some background in basic statistics and probability will be needed to understand the material in these posts. Also, implementations will be done using R, but everything I demonstrate could also be done using Python (and many other languages, if you wish!).

## Setting the Stage for Causal Inference: Randomized Experiments

To understand causal inference and its applications to observational studies, it can be helpful to first understand randomized experiments. We will use the notation similar to that developed by Rosenbaum in his book *Design of Observational Studies* (2010). Suppose we are interested in studying the effects of a new drug used to treat an often fatal illness. There are currently $$I = 100$$ patients with this illness in the hospital and we want to design a randomized experiment to see whether the novel drug is more effective than current standard treatment methods. From now on, we'll call the novel drug the "treatment" and standard care the "control." We will say that patients that receive the treatment will belong to the treatment group and patients that don't receive the treatment will belong to the control group.

We will assign a number in $$\{1, 2, \ldots, I = 100\}$$ to each patient so we can refer to individual patients more conveniently. The vector of observed covariates for patient $$i$$ will be denoted by $$\mathbf{x}_i$$. For example, $$\mathbf{x}_i$$ might contain information about patient $$i$$'s weight, age, height, sex, and other observed covariates. We can also talk about the unobserved covariates for patient $$i$$, denoted by $$\mathbf{u}_i$$. The unobserved covariates include anything that we didn't record about the patient. For example, $$\mathbf{u}_i$$ might contain information about how many calories patient $$i$$ ate last Tuesday or how many cats live in their home.

To indicate whether a patient belongs to the treatment or control group, we'll use $$Z_1, Z_2, Z_3, \ldots, Z_{100} \in \{0, 1\}$$, where $$Z_i = 1$$ indicates that patient $$i$$ belongs to the treatment group and $$Z_i = 0$$ indicates that patient $$i$$ belongs to the control group.

We will need a way to refer to each patient's response. Let $$r_i$$ denote the response from patient $$i$$, where $$r_i = 0$$ indicates that the patient dies and $$r_i = 1$$ indicates that the patient survives. We can also refer to each patient's response specifically when they are given treatment or control with $$r_{Ci}$$ denoting patient $$i$$'s response to control and $$r_{Ti}$$ denoting patient $$i$$'s response to treatment. When a patient $$i$$ is assigned to the control group, i.e., $$Z_i = 0$$, we will know $$r_{Ci}$$ with certainty but we will never know $$r_{Ti}$$ (and vice versa when $$Z_i = 1$$) because we cannot go back in time, change patient $$i$$'s assignment, and observe the other response.

In a randomized experiment, we have complete control over the mechanism of treatment assignment. One straight forward approach is to assign treatment by flipping a fair coin once for each patient. If the coin lands on heads on the $$i^{th}$$ flip, then patient $$i$$ is assigned to the treatment group and this sets $$Z_i = 1$$. If the coin lands on tails, then we assign patient $$i$$ to the control group and this sets $$Z_i = 0$$. This is to say, $$\pi_i = \mathbb{P}(Z_i = 1) = \mathbb{P}(Z_i = 0) = \frac{1}{2}$$, where $$\pi_i$$ is the probability of patient $$i$$ being assigned to the treatment group given the observed covariates, unobserved covariates, response to control, and response to treatment, i.e., $$\pi_i = \mathbb{P}(Z_i = 1 \vert \mathbf{x}_i, \mathbf{u}_i, r_{Ti}, r_{Ci})$$. 

An important characteristic of randomized experiments is that, even though $$\pi_i$$ is defined as a probability conditioned on many things we can never see (like unobserved covariates or the response to the opposite treatment assignment), this doesn't materially impact the experiment because we assign treatment randomly. That is to say, in a randomized experiment, $$\pi_i$$ is independent of $$\mathbf{x}_i, \mathbf{u}_i, r_{Ti}$$, and $$r_{Ci}$$ so that

$$\pi_i = \mathbb{P}(Z_i = 1 | \mathbf{x}_i, \mathbf{u}_i, r_{Ti}, r_{Ci}) = \mathbb{P}(Z_i = 1).$$

### Covariate Balancing in Randomized Experiments

A convenient property arises in randomized experiments due to the randomization of treatment assignments. Since $$\pi_i \perp \mathbf{x}_i, \mathbf{u}_i, r_{Ti}, r_{Ci}$$ simply because the researcher randomly assigns treatment to each subject, randomization tends to produce good covariate balance across the control and treatment groups. To illustrate this property, we can consider a simulated example using the iris data set. 

```r
library(dplyr)

# Set the random seed so the example can easily be reproduced
set.seed(67189)

# Load in iris
data(iris)

# Generate the subject numbers {1, 2,..., 100}
iris$subject_num <- seq(nrow(iris))

# Assign treatment to each subject with equal probability
iris$treatment_ind <- rbinom(nrow(iris), 1, 0.5)

# Remove the Species variable -- this could be an example of a 
# variable that would be better to stratify on
iris <- iris %>% select(-Species)

# Retrieve means of the covariates across the treatment and 
# control group
results <- 
  iris %>% 
  group_by(treatment_ind) %>% 
  summarise(Sepal.Length = mean(Sepal.Length), 
            Sepal.Width = mean(Sepal.Width),
            Petal.Length = mean(Petal.Length), 
            Petal.Width = mean(Petal.Width))
```

On inspection of the resulting covariate balance table, we can see that complete randomization of treatment assignment balances the covariates across the treatment and control groups very well.

<table><caption>Table 1: Iris Covariate Balance, Equally Likely Assignments</caption><tr><th>Group</th><th>Sepal.Length</th><th>Sepal.Width</th><th>Petal.Length</th><th>Petal.Width</th></tr><tr><th>Control</th><td>5.880000</td><td>3.034286</td><td>3.850000</td><td>1.232857</td></tr><tr><th>Treated</th><td>5.811250</td><td>3.077500</td><td>3.677500</td><td>1.170000</td></tr></table>

Compare this to the resulting table generated by reproducing this exact example except with $$\pi_i = 0.15$$.

<table><caption>Table 2: Iris Covariate Balance, Skewed Assignments</caption><tr><th>Group</th><th>Sepal.Length</th><th>Sepal.Width</th><th>Petal.Length</th><th>Petal.Width</th></tr><tr><th>Control</th><td>5.878626</td><td>3.056488</td><td>3.790076</td><td>1.208397</td></tr><tr><th>Treated</th><td>5.600000</td><td>3.063158</td><td>3.536842</td><td>1.136842</td></tr></table>

Although not as balanced as when $$\pi_i = 0.5$$, the covariates are still quite balanced despite a largely biased treatment assignment mechanism. The reason that this is happening comes back to the randomized nature of the treatment assignment and, most importantly, the independence of each $$\pi_i$$ from the subject's covariates and potential responses. 

It becomes more difficult to produce acceptable covariate balance the further we push down $$\pi_i$$. Observe how the balance degrades when we push $$\pi_i$$ down to $$0.05$$.

<table><caption>Table 3: Iris Covariate Balance, Very Skewed Assignments</caption><tr><th>Group</th><th>Sepal.Length</th><th>Sepal.Width</th><th>Petal.Length</th><th>Petal.Width</th></tr><tr><th>Control</th><td>5.834931</td><td>3.062329</td><td>3.717123</td><td>1.178767</td></tr><tr><th>Treated</th><td>6.150000</td><td>2.875000</td><td>5.250000</td><td>1.950000</td></tr></table>

This becomes less of a problem as $$I$$ grows larger; in this example, $$I = 150$$ and so when we set $$\pi_i = 0.05$$, we can only expect about $$7$$ or $$8$$ subjects to be assigned to the treatment group which makes poor quality covariate balancing more likely.

You may have noticed that in describing the balancing property of randomization, we did not distinguish between balancing the observed versus unobserved covariates. This is because randomization guarantees balance across both. To see this, suppose that we gathered the iris data but chose not to record petal width. This decision would not have changed the balance of the petal width covariate because we did not vary treatment assignment probability based on petal width. In fact, we did not vary treatment assignment probability on any covariate. A flower with extremely wide petals is just as likely to be assigned to the treatment group as a flower with narrow petals and this is regardless of whether the researcher chooses to observe or record this information. Given some other covariate that was not measured, say, the age of the iris plant at the time the measurements were taken, we can still be confident that it would be well balanced across the treatment and control groups. One of the main challenges in observational studies is that unobserved covariate balance is no longer guaranteed and we will eventually need strategies to work around this problem.

### Fisher's Sharp Hypothesis of No Effect

The primary motivation of a randomized experiment is to identify measurable treatment effects when they exist. This motivation informs the idea of a null hypothesis often used in randomized experiments known as Fisher's Sharp Hypothesis of No Effect. The hypothesis says that if the treatment effect does not exist, then $$r_{Ci} = r_{Ti}$$ for each patient $$i$$. In words, this would mean that each patient's response is the same regardless of which group they were assigned to, implying no treatment effect. As discussed earlier, we can never know whether this is the case for a particular patient $$i$$ because if we observe $$r_{Ci}$$, we can never observe $$r_{Ti}$$ (and vice versa). The question now becomes: how can we determine whether the results of the experiment point to Fisher's hypothesis being false?

Although we cannot say anything about any particular patient $$i$$, in aggregate, we can often say a lot more about all $$I$$ patients. Suppose that in our trial, every patient in the treatment group survives but every patient in the control group dies. Even though we cannot say with certainty how each individual patient would have responded had they been placed in the opposite group, in aggregate, it should not be difficult to convince ourselves and others that in this instance, Fisher's hypothesis is extremely unlikely to be true and the drug we're testing is a highly effective, life saving treatment.

Outside of experiments in the hard sciences (e.g., physics, chemistry), it is exceedingly rare for an experiment to produce such definitive results. Part of this is because, despite how carefully we design experiments, there is almost certainly variation among the observed and unobserved covariates associated with each subject. In the trial we've been using as an example, we do not have $$I = 100$$ copies of the exact same person; we have $$I = 100$$ unique individuals. Some are women and others are not, some are short and others are not, some are old, and so on. Different individuals may respond to the treatment in different ways, and this generally means that the results of experiments are rarely completely cut and dry.

## Stepping Toward Observational Studies

There are a few important differences between randomized experiments and observational studies. Observational data are often collected by someone other than the researcher and well before the idea of the study has been conceived. Randomized experiments are more structured in that the data are collected at one or several carefully pre-designated times because the researcher has complete control over the data collection process.

Another crucially important difference that was briefly touched on earlier lies in the mechanism of treatment assignment. In a randomized experiment, the researcher has complete control over this mechanism. In an observational study, treatment assignment and administration both could have occurred at different times for different sets of subjects at different degrees. The exact mechanism through which treatment was assigned could be completely unknowable. All of this is to say that in an observational study, the researcher has no control over which subject was assigned to which group or how the treatment was administered.

The primary goal in the design of an observational study is to replicate the conditions of a randomized experiment as closely as possible so that the same convenient properties of randomized experiments can be used to look for measurable causal treatment effects. The treatment assignment mechanism presents a major obstacle to this goal. We now examine a phenomenon called Simpson's Paradox to illustrate this obstacle.

### Simpson's Paradox

To describe Simpson's Paradox, we will recreate an example given by Rosenbaum in his book *Observation & Experiment*. Consider the same trial that we've used as an example except in this case, we have $$I = 400,000$$ patients separated into four distinct strata based on age (older/younger) and biological sex (male/female). Each stratum contains $$100,000$$ patients. Of crucial importance in this example is that we will vary the probability of treatment $$\pi_i$$ based on age; patients in the older strata will have a probability of treatment $$\pi_i = 0.8$$ whereas patients in the younger strata will have $$\pi_i = 0.2$$. Since we are simulating this example, let's also suppose that Fisher's hypothesis of no effect is true in this instance so that regardless of treatment, $$40\%$$ of older men, $$30\%$$ of older women, $$20\%$$ of younger men, and $$10\%$$ of younger women die.

We can recreate this simulated example in R:

```r
# Set the random seed so the example can be easily reproduced
set.seed(17689)

# Generate the patient numbers {1, 2,..., 400,000}
patient <- seq(400000)

# Assign 100,000 patients to each stratum; OM for Older Men, YW 
# for Younger Women, and so on
stratum <- c(rep("OM", 100000), rep("OW", 100000), 
             rep("YM", 100000), rep("YW", 100000))

# Assign treatment; the first 200,000 patients are older so 
# the probability of treatment is 0.8 and the remaining patients 
# are younger so the probability of treatment is 0.2
treatment <- c(rbinom(200000, 1, 0.8), rbinom(200000, 1, 0.2))

# Assign responses based on the probability of death described above
outcome <- c(rbinom(100000, 1, 0.4), rbinom(100000, 1, 0.3), 
             rbinom(100000, 1, 0.2), rbinom(100000, 1, 0.1))

```

Each row of the resulting data frame represents one patient and tells us which stratum the patient belongs to, whether the patient received the treatment being studied, and whether the patient survived. These simulated data are summarized in the following table.

<table>
<caption>Table 4: Simulated Pseudo-Randomized Experiment Results, Stratified</caption>
  <tr>
    <th>Group</th>
    <th>Dead</th>
    <th>Alive</th>
    <th>Total</th>
    <th>Mortality Rate (%)</th>
  </tr>
  <tr>
    <th colspan=5>Stratum 1: Older Men</th>
  </tr>
  <tr>
    <td>Control</td>
    <td>7925</td>
    <td>12064</td>
    <td>19989</td>
    <td>39.6</td>
  </tr>
  <tr>
    <td>Treated</td>
    <td>31649</td>
    <td>48362</td>
    <td>80011</td>
    <td>39.6</td>
  </tr>
  <tr>
    <th colspan=5>Stratum 2: Older Women</th>
  </tr>
  <tr>
    <td>Control</td>
    <td>6076</td>
    <td>13727</td>
    <td>19803</td>
    <td>30.7</td>
  </tr>
  <tr>
    <td>Treated</td>
    <td>24199</td>
    <td>55998</td>
    <td>80197</td>
    <td>30.2</td>
  </tr>
  <tr>
      <th colspan=5>Stratum 3: Younger Men</th>
  </tr>
  <tr>
    <td>Control</td>
    <td>16180</td>
    <td>63736</td>
    <td>79916></td>
    <td>20.2</td>
  </tr>
  <tr>
    <td>Treated</td>
    <td>3929</td>
    <td>16155</td>
    <td>20084</td>
    <td>19.6</td>
  </tr>
  <tr>
      <th colspan=5>Stratum 4: Younger Women</th>
  </tr>
    <tr>
    <td>Control</td>
    <td>7992</td>
    <td>71939</td>
    <td>79931</td>
    <td>10.0</td>
  </tr>
  <tr>
    <td>Treated</td>
    <td>1955</td>
    <td>18114</td>
    <td>20069</td>
    <td>9.7</td>
  </tr>
</table>

We already know that Fisher's hypothesis of no effect is true because we simulated this example precisely so it would be true. In each stratum, the mortality rates across the treated and control groups are very similar. Notice how this changes though when the results are all grouped together.

<table>
<caption>Table 5: Simulated Pseudo-Randomized Experiment Results, Unstratified</caption>
  <tr>
    <th>Group</th>
    <th>Dead</th>
    <th>Alive</th>
    <th>Total</th>
    <th>Mortality Rate (%)</th>
  </tr>
  <tr>
    <td>Control</td>
    <td>38173</td>
    <td>161466</td>
    <td>199639</td>
    <td>19.1</td>
  </tr>
  <tr>
    <td>Treated</td>
    <td>61732</td>
    <td>138629</td>
    <td>200361</td>
    <td>30.8</td>
  </tr>
</table>

This table paints a very different picture. The treatment appears to be causing a much higher mortality rate than the control, but we know that this is false because of the way we simulated this example. This is Simpson's Paradox, a phenomenon where some type of relationship appears in groups of data (in this case, the strata) but then disappears or reverses when the data are grouped together.

In randomized experiments, the researcher has complete control over each $$\pi_i$$. If it makes sense within the context of the problem to vary $$\pi_i$$ based on a subject's covariates, the researcher is free to make this decision and interpret results via stratification. Observational studies do not have this luxury. Researchers often analyze the data long after they were collected and have no control over the treatment assignment mechanism. Failing to account for treatment assignment probability in an observational study can therefore produce results that are in direct conflict with what is truly happening in reality.

The aggregated table above would only have a meaningful interpretation in a completely randomized trial where the $$\pi_i$$ are equal across all $$400,000$$ patients. In observational studies, there is a need for a tool that can control for this problem of unknown probability of treatment assignment.

## What's Coming Next

So far, we've discussed how randomized experiments work and how they compare with observational studies. We've started to touch on the main difficulty in analyzing observational data, which is that we cannot assume that unobserved covariates are balanced across the treatment and control groups. If we make this assumption, we might end up with a situation like Simpson's Paradox, and we may end up saying that a relationship is there when it doesn't really exist. 

In the next post, we'll start talking about pair matched designs experimental designs. We'll discuss pair matching algorithms that use a helpful tool called a propensity score that will make quick work of pairing like subjects together to minimize unobserved bias as much as possible. Stay tuned!