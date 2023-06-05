---
layout: post
cover: 'assets/images/covers/nebula2-cover.jpeg'
navigation: True
title: Introduction to Causal Inference - Part 2
date: 2023-05-29 7:00:00
tags: [Causal Inference]
subclass: 'post tag-causalinference'
logo: 'assets/images/logo.png'
author: Austin
categories:
---

In [part 1](https://austinpurdie.com/causal-inference-part-1) of this series on basic causal inference methods, we talked through randomized experiments and how they compare to observational studies. We talked about how the main difference between randomized experiments and observational studies lies in the treatment assignment mechanism. A researcher has complete control over this mechanism in a randomized experiment, and this control guarantees balance across the control and treatment groups for both observed and unobserved covariates.

Observational studies do not have this luxury. Since we do not know the treatment assignment mechanism, we can never know whether a particular unobserved covariate is balanced, and in practice, we will assume that many unobserved covariates are not balanced. This lack of balance leads to an unknowable quantity of bias in our results, and we will need to develop tools to address this issue.

- [Introduction](#introduction)
- [Matched Pair Designs, the Naive Model, and Propensity Scores](#matched-pair-designs-the-naive-model-and-propensity-scores)
- [Detangling the Pair Matching Process](#detangling-the-pair-matching-process)
- [What's Next](#whats-next)

## Introduction

In this post, we'll discuss pair matched experimental designs and how we can use pair matching to analyze observational data. We'll also talk through a method to simplify the pair matching process using propensity scores. In a future post, we'll discuss how to analyze matched pair outcomes with Wilcoxon's Signed Rank Test and how to measure the study design's sensitivity to the unobserved bias mentioned earlier.

## Matched Pair Designs, the Naive Model, and Propensity Scores

In order to address the problem of treatment assignment probability, it will be useful to adopt a pair matched trial design. To balance the $$\pi_i$$, we will want to find several pairs of treated and control subjects that 1) have similar probabilities of treatment assignment and 2) are as similar as possible in terms of the observed covariates. That is, for each pair $$p$$, 

$$\mathbf{x}_{Cp} \thickapprox \mathbf{x}_{Tp} \text{ and } \pi_{Cp} \thickapprox \pi_{Tp}.$$

To explain why this type of design can be useful, we follow Rosenbaum's derivation of the "naive" model from his book *Design of Observational Studies* (2010), beginning with defining the $$\pi_i$$ so that

$$\pi_i = \mathbb{P}(Z_i = 1 | r_{Ti}, r_{Ci}, \mathbf{x}_i, \mathbf{u}_i),$$

$$\mathbb{P}(Z_1 = z_1, \ldots, Z_L = z_L | r_{T1}, r_{C1}, \mathbf{x}_1, \mathbf{u}_1, \ldots, r_{TL}, r_{CL}, \mathbf{x}_L, \mathbf{u}_L)$$

$$\textstyle = \pi_1^{z_1} (1 - \pi_1)^{1 - z_1} \cdots \pi_L^{z_L} (1 - \pi_L)^{1 - z_L} = \prod_{i = 1}^L \pi_i^{z_i} (1 - \pi_i)^{1 - z_i}.$$

What this is saying is that each subject $$i$$ is assigned to the treatment group with probability $$\pi_i$$ independently from the other subjects. Stepping toward pair matching, suppose we could not only find two subjects $$k$$ and $$l$$ such that $$\pi_k = \pi_l$$, but also that exactly one of the subjects was treated so that $$Z_k + Z_l = 1$$. Then using the definition of $$\pi_i$$ from above, we have that

$$
\begin{align*}
  & \mathbb{P}(Z_k = 1, Z_l = 0 | r_{Tk}, r_{Ck}, \mathbf{x}_k, \mathbf{u}_k, r_{Tl}, r_{Cl}, \mathbf{x}_l, \mathbf{u}_l, Z_k + Z_l = 1) \\ \\
  & = \frac{\mathbb{P}(Z_k = 1, Z_l = 0 | r_{Tk}, r_{Ck}, \mathbf{x}_k, \mathbf{u}_k, r_{Tl}, r_{Cl}, \mathbf{x}_l, \mathbf{u}_l)}{\mathbb{P}(Z_k + Z_l = 1 | r_{Tk}, r_{Ck}, \mathbf{x}_k, \mathbf{u}_k, r_{Tl}, r_{Cl}, \mathbf{x}_l, \mathbf{u}_l)} \\ \\
  & = \frac{\pi_l^{1 + 0} (1 - \pi_l)^{(1 - 1) + (1 + 0)}}{\pi_l^{1 + 0} (1 - \pi_l)^{(1 - 1) + (1 - 0)} + \pi_l^{0 + 1} (1 - \pi_l)^{(1 - 0) + (1 - 1)}} \\ \\
  & = \frac{\pi_l (1 - \pi_l)}{\pi_l (1 - \pi_l) + \pi_l (1 - \pi_l) } = \frac{1}{2},
\end{align*}
$$

meaning that if we created several pairs in this way, we could exactly replicate the same completely randomized treatment assignment mechanism discussed in the last post. This would make the inference very straight forward and any of the familiar tools of statistical inference could be used. Unfortunately, in an observational study, we do not know $$\pi_l$$ or $$\pi_k$$. Even worse, we cannot estimate these probabilities since they depend on their unobserved covariates, responses to treatment, and responses to control. Here we introduce the naive model, which assumes that the $$\pi_i$$ only depend on the observed covariates (hence the model's naivety) and that $$\pi_i \in (0, 1)$$ so that every subject has some chance of being assigned treatment. Formally, the naive model says that

$$\pi_i = \mathbb{P}(Z_i = 1 \vert r_{Ti}, r_{Ci}, \mathbf{x}_i, \mathbf{u}_i) = \mathbb{P}(Z_i = 1 \vert \mathbf{x}_i)$$

$$\text{ and } 0 < \pi_i < 1, i = 1, 2, \ldots, L \text{ with}$$

$$\textstyle \mathbb{P}(Z_1 = z_1, \ldots, Z_L = z_L \vert r_{T1}, r_{C1}, \mathbf{x}_1, \mathbf{u}_1, \ldots, r_{TL}, r_{CL}, \mathbf{x}_L, \mathbf{u}_L) = \prod_{i = 1}^L \pi_i^{z_i} (1 - \pi_i)^{1 - z_i}.$$

Rosenbaum calls this set of conditions "strongly ignorable treatment given $$\mathbf{x}$$," meaning that if the naive model is true, we can essentially stop worrying about the treatment assignment mechanism and treat the study as if it were a randomized experiment. Additionally, it would mean that subject $$k$$ and subject $$l$$ would be ideally matched if $$\mathbf{x}_k = \mathbf{x}_l$$.

Exact matching across one or two covariates presents some challenges but can be manageable if there are enough subjects to draw from. The pair matching problem becomes much more difficult when dealing with five, ten, or dozens of covariates. For example, a configuration of $$20$$ binary covariates is $$1$$ out of $$2^{20} = 1,048,576$$ possible combinations. The problem quickly becomes intractable when we are dealing with continuous covariates. To address this issue, we can make use of propensity scores, denoted by $$e(\mathbf{x})$$. The propensity score is the conditional probability of treatment given the vector of observed covariates $$\mathbf{x}$$:

$$e(\mathbf{x}) = \mathbb{P}(Z = 1 | \mathbf{x}).$$

Notice that in the naive model $$e(\mathbf{x}_i) = \pi_i$$. While similar to $$\pi_i$$ in that the propensity score is a quantity that we cannot derive exactly, $$\pi_i$$ differs from the propensity score in a critical way because we can compute an estimate $$\widehat{e}(\mathbf{x})$$ of the propensity score using the vector of observed covariates $$\mathbf{x}$$. This is a direct consequence of the propensity score's independence of $$\mathbf{u}_i, r_{Ci}$$, and $$r_{Ti}$$.

The propensity score is often estimated using a logistic regression model but other GLMs and modeling techniques can be used. Propensity scores have a useful balancing property such that when pairs are balanced on the propensity score, they tend to also be balanced on the observed covariates. Instead of balancing on several individual covariates, we can choose to balance on one covariate, the estimated propensity score. In doing this, we will ideally achieve an acceptable level of balance across the observed covariates $$\mathbf{x}$$ without explicitly balancing them.

Once estimates of each subject's propensity score have been obtained, we can begin the process of identifying ideal matched pairs.

## Detangling the Pair Matching Process

Suppose we have $$L$$ subjects in an observational data set and we wish to construct a pair matched observational study using $$2I$$ of the $$L$$ subjects. Using logistic regression, we compute estimates of each of the $$L$$ subject's propensity scores and now we're ready to match. How exactly do we go about finding ideal matches?

One of the primary challenges of the pair matching process is that as pairs are formed, competition for the remaining control or treated subjects increases. Just because one pair seems optimal at the beginning of the process doesn't mean it will be part of the most optimal matching for the entire data set.

A common approach to pair matching is to first identify all possible matched pairs using a caliper, meaning that pairs between the treated subject $$k$$ and the control subject $$l$$ will be forbidden when, given a caliper $$c \in \mathbb{R}$$,

$$|\widehat{e}(\mathbf{x}_k) - \widehat{e}(\mathbf{x}_l)| \geq c.$$

This first step guarantees a predetermined degree of closeness in the estimates of the propensity scores across each of the matched pairs. Rosenbaum suggests starting with a caliper equal to $$20\%$$ of the standard deviation of the $$L$$ propensity score estimates.

Once pairs outside of the caliper restriction have been excluded, the optimal pair for each treated subject can be identified by finding the pair which minimizes some distance function. A common choice is the Mahalanobis distance. 

Developed by Prasanta Chandra Mahalanobis in the 1930s, the Mahalanobis distance is a measure of the distance between two points in $$\mathbb{R}^n$$ from some distribution $$\mathcal{Q}$$ with covariance matrix $$\boldsymbol\Sigma$$. It differs from the Euclidean distance because it weights the distance between covariates less heavily when they are highly correlated. Formally, the Mahalanobis distance between two points $$\mathbf{x}_k$$ and $$\mathbf{x}_l$$ is given by

$$d_m(\mathbf{x}_k, \mathbf{x}_l; \mathcal{Q}) = (\mathbf{x}_k - \mathbf{x}_l)^\top \boldsymbol\Sigma^{-1}(\mathbf{x}_k - \mathbf{x}_l).$$

In an observational study, we will not know the distribution $$\mathcal{Q}$$ explicitly, but we can easily obtain an estimate of the Mahalanobis distance between any two observations by using the sample covariance matrix $$\widehat{\boldsymbol\Sigma}$$:

$$\widehat{d_m}(\mathbf{x}_k, \mathbf{x}_l; \mathcal{Q}) = (\mathbf{x}_k - \mathbf{x}_l)^\top \widehat{\boldsymbol\Sigma}^{-1}(\mathbf{x}_k - \mathbf{x}_l).$$

Rosenbaum points out some problems with using the Mahalanobis distance as the distance function for pair matching. It can produce misleading distances if there are strong outliers in some of the covariates or if the standard deviation of a covariate is particularly large. In these cases, the Mahalanobis distance will tend to deprioritize or completely ignore these covariates.

A modified version of the Mahalanobis distance can be can be used with data that are not normally distributed. This approach is called the rank-based Mahalanobis distance and is obtained through the following process:

- Replace each covariate for each subject with its ascending rank, average ranks for ties; for example, if there are $$L = 4$$ subjects with ages $$31$$, $$19$$, $$47$$, and $$19$$, these would be replaced with $$3$$, $$1.5$$, $$4$$, and $$1.5$$ respectively.
- Premultiply and postmultiply the covariance matrix of the ranks $$\boldsymbol\Sigma_{\text{rank}}$$ by a diagonal matrix $$\mathbf{D}$$ whose diagonal elements are the ratios of the standard deviation of untied ranks to the standard deviations of the tied ranks for each covariate.
- Compute the Mahalanobis distance using $$\mathbf{D}\boldsymbol\Sigma_{\text{rank}}\mathbf{D}$$ instead of $$\boldsymbol\Sigma$$.

Rosenbaum says that the ranking step helps to reduce the influence of outliers and the adjusted covariance matrix helps to prevent covariates with low variance from being weighted too heavily. Combining the rank-based Mahalanobis distance with a caliper on the propensity score is a good strategy to solve the pair matching problem. That is, the pair matching process will be carried out in two steps. First, we will obtain the set of every possible match that meets the caliper requirement. Second, from this set of possible matches, we will select the configuration that minimizes the rank-based Mahalanobis distance. In doing this, we not only ensure that each treated subject and its paired control subject will have similar estimated propensity scores, we also find pairs of points that "look" comparable.

The R package *optmatch* has been designed to carry out this process. The packages offers a selection of different pair matching algorithms to choose from. The default option is the cycle canceling algorithm, which was developed for use on the minimum-cost flow problem. The minimum-cost flow problem is an optimization problem that is used to find the least costly way of transporting flow through a flow network. An example of this problem is a package delivery company finding the most optimal delivery routes such that the smallest amount of fuel is consumed. To apply this to the ideal matching problem, we set the cost function to the rank-based Mahalanobis distance and represent the "delivery routes" with the individual potential connections between treated and control subjects. Routes that violate the caliper restriction are immediately removed from consideration. Then, from the remaining possibilities, the rank-based Mahalanobis distance is used to arrive at the most ideal set of matches. The implementation of *optmatch* to complete the pair matching process in R will be demonstrated in the last post where we'll put together all of the theory to design an observational study from scratch.

## What's Next

In the next post, we'll take a deep dive into Wilcoxon's Signed Rank Test. We'll define its null and alternative hypotheses, the test procedure, and we'll derive the the normal approximation of its $$p$$-values. We'll also discuss a dose weighted modification of Wilcoxon's test that is more suitable for observational data where treatment is administered in varying doses.