# project-work

# Approaches
We used multiple approaches to solve the gold collection problem.

## Merge Approach

The merge approach is a constructive heuristic that builds efficient multi-stop routes by combining single-city trips.

### Algorithm Overview

1. **Single-City Trip Generation**
   - First, compute the optimal round trip for each city individually (depot → city → depot)
   - For the outbound journey (unloaded), use Dijkstra's algorithm to find the shortest path
   - For the return journey (loaded with gold), use A* search with a weighted heuristic that accounts for the gold being carried
   - Each trip's cost is calculated using the weight-dependent cost function: `cost = dist + (dist × α × weight)^β`

2. **Trip Sorting**
   - Sort all single-city trips by path length in descending order
   - Longer trips are processed first because they offer more opportunities for merging

3. **Greedy Merging**
   - For each main trip, examine cities that lie on its return path
   - For each candidate city on the return path:
     - Calculate the **marginal cost**: the additional cost of picking up that city's gold versus continuing with current weight
     - Compare marginal cost with the standalone cost of doing that city separately
     - If `marginal_cost < standalone_cost`, merge the city into the current trip
   - Mark merged cities as "excluded" to prevent them from being used in other trips

4. **Path Optimization ($\beta > 1$ only)**
   - When $\beta > 1$ we can apply another optimization explained in detail below

### Key Features

- **A\* Heuristic**: Uses Euclidean distance as a lower bound for the weighted travel cost
- **Distance Caching**: Pre-computes all edge distances in a path for efficient cost recalculation
- **Greedy Selection**: Makes locally optimal merge decisions without backtracking
- **Non-worse performance**: The approach of the algorithm guarantees for all instances that the path found is at least as good as the baseline

### Complexity

- Single-city trip generation: $O(n \times E \log V)$ where $n$ is the number of cities
- Merging phase: $O(n^2 \times L)$ where $L$ is the average path length
- Overall: Polynomial time, making it suitable for large instances

## Beta optimization
When $\beta > 1$ we can exploit the weights' non linearities to subdivide each round trip in $N$ trips, each one picking $\frac{w}{N}$ (where $w$ is the total gold picked)

To explain the approach we first consider a simpler case, to get used to the notation, then extend to complex paths.

### Simple case
Let's consider first a single round trip where the gold is picked from only one city.
Let's denote 
- $C_{\text{go}}$ as the cost to get to target city (where gold is picked) 
    - $P_{\text{go}}$ is such path represented as pairs `[from, to]` representing pairs of each step
- $C_{return}$ as the cost to return from target city to 
    - $P_{\text{return}}$ as explained before
    - $C_{\text{ret\_s}}$ is the static part, depending only on geoemtric distance
    - $C_{\text{ret}}(w)$ is the part depending on weight carried
$$
    C_{\text{go}} = \sum_{(i,j) \in P_{\text{go}}} d_{i,j} \\
    C_{\text{return}} = \sum_{(i,j) \in P_{\text{return}}} d_{i,j} + (d_{i,j} \alpha w)^\beta = 
    \sum_{(i,j) \in P_{\text{return}}} d_{i,j} + 
    \sum_{(i,j) \in P_{\text{return}}} (d_{i,j} \alpha w)^\beta =  C_{\text{ret\_s}} + C_{\text{ret}}(w)
$$

Finally the total cost of the round trip can be computed ad
$$
    C = C_{\text{go}} + C_{\text{ret\_s}} + C_{\text{ret}}(w)
$$

Now we consider the cost picking $\frac{w}{N}$ gold at each round-trip, repeated $N$ times

$$
    C(N) = N \left( C_{\text{go}} + C_{\text{ret\_s}} + C_{\text{ret}} \left(\frac{w}{N} \right) \right )
$$

We want to find $N^*$ such that it minimizes the cost.

To find such value, we treat $N$ as a continous variable and differentiate w.r.t. $N$
$$
    C'(N) = C_{\text{go}} + C_{\text{ret\_s}} + (1 - \beta)N^{-\beta}(\alpha w)^\beta \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta = 0 
$$

Solving the equation yields

$$
    N^* = \alpha w \left ( \frac{\beta - 1} {C_{\text{ret\_s}} + C_{\text{go}} } \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta \right ) ^ {\frac{1}{\beta}}
$$

**Note**
- Given the scale of values of $w$ in the context, it is almost always beneficial to split a path into multiple subpaths
- From the formula, it is clear the constraint of $\beta > 1$, otherwise the problem becomes linear or sublinear and the exploit is never beneficial

### General case
In the general case we consider a single round trip in which gold can be picked from multiple cities in the same trip.

We subdivide the path into subpaths where gold carried is constant.

For example consider the path `[0, 2, 4, 5(G), 3, 6(G), 2, 0]`, we can subdivide it into 3 subpaths carrying fixed gold
$$
    S = \{ \{ 0, 2, 4, 5\}, \{ 5, 3, 6 \}, \{ 6, 2, 0 \} \}
$$
Carrying respectively `[0, gold(5), gold(5)+gold(6)]` total gold

Considering the cost from a single path to introduce some notation
$$
    C_s = \sum_{(i,j) \in s} d_{i,j} + (d_{i,j} w_s \alpha)^\beta = C_{\text{s\_static}} + C_{s\_\text{dyn}}(w_s)
$$

We can express the total cost of a single round trip as 
$$
    C = \sum_{s \in S} C_s = 
    \sum_{s \in S} \sum_{(i,j) \in S} d_{i,j} + \sum_{s \in S} \sum_{(i,j) \in S} (\alpha w_s d_{i,j})^\beta =
    \sum_{s \in S} C_{\text{s\_static}} + C_{s\_\text{dyn}}(w_s)
$$

As before we now consider splitting the path in $N$ identical trips, picking $\frac{w}{N}$ gold each

$$
    C(N) = N \left( \sum_{s \in S} C_{\text{s\_static}} + C_{s\_\text{dyn}}\left(\frac{w_s}{N}\right) \right) 
    = N \sum_s C_{s\_\text{static}} + \alpha^\beta N^{-\beta+1} \sum_s w_s^\beta \sum_{(i,j) \in s} d_{i,j}^\beta
$$

To simplify a bit the notation we introduce

$$
    C_{s\_\text{beta\_static}} (\beta) = \sum_{(i,j) \in s} d_{i,j}^\beta
$$

Getting

$$
    C(N)
    = N \sum_s C_{s\_\text{static}} + \alpha^\beta N^{-\beta+1} \sum_s w_s^\beta C_{s\_\text{beta\_static}} (\beta)
$$

Once again, differentiating w.r.t. $N$ and setting the derivative to $0$ yields

$$
    C'(N) = \sum_s C_{s\_\text{static}} + (1 - \beta) \alpha^\beta N^{-\beta} \sum_s w_s^\beta C_{s\_\text{beta\_static}} (\beta) = 0
$$

Solving for optimal $N^*$ gives

$$
    N^* = \left ( 
        \alpha^\beta (\beta - 1) \frac{\sum_s w_s^\beta C_{s\_\text{beta\_static}} (\beta)}{\sum_s C_{s\_\text{static}}}
    \right)^{\beta^{-1}}
$$

### Implementation Strategy

The derived formula for $N^*$ can be applied independently to each round trip in the solution:

1. **Decomposition**: The complete solution (visiting all cities) is naturally decomposed into multiple round trips by the merge algorithm
2. **Independent Optimization**: Each round trip is optimized separately using its own $N^*$ calculated from the formula above
3. **Reconstruction**: The optimized round trips are concatenated to form the final solution

This approach is valid because:
- Each round trip is independent (starts and ends at depot)
- The optimization formula depends only on the trip's internal structure (distances, weights, pickup sequence). The result depends on the importance of the static costs
- No interaction exists between different trips that would prevent independent optimization

### Key Advantages

- **Analytical Solution**: Closed-form formula with no hyperparameters or iterative optimization
- **Computational Efficiency**: $O(L)$ per trip, where $L$ is the trip length
- **(Almost) Guaranteed Improvement**: When $\beta > 1$, splitting almost always reduces cost (superlinear cost function)
- **Exact Optimum**: The continuous relaxation provides the theoretically optimal split count (rounded to nearest integer)