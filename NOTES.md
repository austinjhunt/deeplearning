# Notes

These are just some notes about different deep learning topics compiled from various resources while practicing the hands-on coding.

## [Sigmoid Functions](https://www.youtube.com/watch?v=Aj7O9qRNJPY)

- Example: predict high school drop outs
- Model outputs unbounded score, -inf, +inf
  - lower score -> more evidence student will not drop out
  - higher score -> more evidence student will drop out
- Goal: convert score (-inf, +inf) to a probability P (between 0 and 1) for interpretability
- Say you have some finite set of students. This means you have a defined upper and lower bound for your output score. Assume -10 and +10.
- Draw graph: -10 to +10 for x axis, 0 to 1 for y. Draw straight line from -10, 0 to 10,1 on graph. Could assume that is your map from the scores to the probabilities. This isn't the best choice. Why?
  - Issues:
    - what if score for a given student is > 10 or score < -10? linear model would map it out of bounds. (over 1 or below 0 probability)
    - rate of change. linear function doesn't capture accurately the rate of change.
      - assume score is 0, which maps to probability of 1/2. 50% chance student will drop out.
        - if score goes from 0 to 1, probability of drop out spikes, doesn't just increase linearly.
        - if score goes from 0 to -1, probability of drop out drops sharply, doesn't just decrease linearly.
      - now assume score is 9 and goes to 10. not a significant change in probability of drop out, relatively speaking. rate of change drops as you get toward boundaries of your x axis

### The math

$$ p(s) = \frac{1}{1 + e^{-s}} = (1 + e^{-s})^{-1}$$
last part of equation is helpful for obtaining derivative

p(s) = probability given the score; e = euler's number (2.7...); s = score

- assume score is positive infinity. $\frac{1}{1+0} = 1$, i.e. probability of drop out = 100%
- assume score is negative infinity. $\frac{1}{\infty} = 0$, i.e. probability of drop out = 0
- assume score is 0, no information. $\frac{1}{1+1} = \frac{1}{2}$ probability of drop out = 50%

### Derivative of the Sigmoid

We care about this because it's used to calculate loss functions, which are used to train machine learning models.
$$\frac{dp}{ds} = \frac{e^{-s}}{(1+e^{-s})^2}$$

really nice derivative property of sigmoid: we can split this into multiplication of two terms

$$
\underbrace{\frac{1}{1+e^{-s}}}_{p(s)} * \underbrace{\frac{e^{-s}}{1+e^{-s}}}_{1-p(s)}
$$

derivative of sigmoid with respect to s/the score is p \* 1-p

## [Softmax Loss Function](https://www.youtube.com/watch?v=8ps_JEW42xs)

- comes up a lot in ML, especially with neural networks
- multidimensional, multiple classes that you are trying to predict
- example: predict what major a high schooler will choose when they first enter college
- model outputs N scores: $S=[s_1, s_2, ... s_n] $ where $s_i \in (-\infty, +\infty)$
  - each score corresponds to a major/class
  - higher score means more evidence that student will choose that corresponding major
  - lower score means lower confidence/chance that student will choose that major
- Goal: transform scores (-inf, +inf) into a vector of probabilities ($p_i$)
- If we're going to transform to probabilities, two things need to be true:
  - each $p_i$ needs to be between 0 and 1, i.e. $0 \leq p_i \leq 1$, and
  - $\sum_i^n{p_i}=1$
- idea 1 (not a great idea, but starting from basics):
  $$
  p_i = \frac{s_i}{\sum_j^n{s_j}}
  $$
  - assume three majors, and the scores are 0, 1, and 2
  - ISSUE
    $$
    \begin{align}
    \begin{bmatrix}
           0 \\
           1 \\
           2 \\
    \end{bmatrix} \implies
    \begin{bmatrix}
            0 \\
            \frac{1}{3} \\
            \frac{2}{3} \\
    \end{bmatrix}
    \end{align}
    $$
    right off the bat, kind of weird that we're giving first major a zero probability already.
    now, assume the scores are instead 100, 101, 102.

$$
\begin{align}
\begin{bmatrix}
       100 \\
       101 \\
       102 \\
\end{bmatrix} \implies
\begin{bmatrix}
        .33 \\
        .33 \\
        .34 \\
\end{bmatrix}
\end{align}
$$

The difference is pretty much the same between any pair of elements, just boosted by 100. But the probabilities are drastically different.

**We would like our final probabilities to be invariant, regardless of adding/subtracting constants from the scores.**

Adding a constant to your score vector should not drastically change your probability vector. That's where softmax comes in.

Consider:

$$
p_i = \frac{e^{s_i}}{\sum_j^n{e^{s_j}}}
$$

fixes a ton of issues. still satisfies probability assumptions/requirements.
don't have to worry about scores being positive or negative because e to the power of anything is positive.

also, adding a constant doesn't affect probability :) see below.

$$
p_i' = \frac{
    e^{s_i + C}
}{
    \sum_j^n{e^{s_j + C}}
} =  \frac{
    e^{s_i}e^{C}
}{
    \sum_j^n{e^{s_j}e^C}
} =  \frac{
    e^{s_i}
}{
    \sum_j^n{e^{s_j}}
} = p_i
$$

## Partial Derivatives of Softmax Function

_if you change the score of the major, how does the probability of the major being chosen change?_

$$
\frac{\partial{p_i}}{\partial{s_i}} = \frac{
    (\sum)e^{s_i} - [e^{s_i}]^2
}{
(\sum)^2
} = p_i - p_i^2 = p_i * (1 - p_i)
$$

This makes sense because if probability of student choosing major X was already very high, then the score changing just a little bit will not affect probability.
BUT if probability of student choosing major X is equal to probabilities for all other majors, and the score changes a little bit for X, then the probability will change significantly.

### Cross terms derivatives

What's the derivative of $p_i$ with respect to $s_j$? Assuming i and j are different. We're looking at relationships; how does the probability of the math major being chosen change if you change the score of the history major? Well, if you increase the score of another major, then the change for the major in question should be negative since increasing the probability of one should decrease the probability of the other.

$$
\frac{\partial{p_i}}{\partial{s_j}} = -p_ip_j
$$

where $p_i$ is probability of choosing major i, $p_j$ is probability of choosing j

if $p_i$ and $p_j$ are both $\frac{1}{2}$, then all other majors have probability of 0, and choice is just between majors i and j. So if you nudge this a little bit by slightly increasing $s_i$, then $p_i$ will drastically increase and $p_j$ will drastically drop.
