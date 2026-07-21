# Chapter 2 Related Work

## Main Argument

Existing TRS:
Single POI recommendation (87% of existing systems output flat ranked lists)
        │
        ▼
CBF / CF Algorithms:
Strong personalization, high transparency, and explainability as baseline algorithms
        │
        ▼
Itinerary Systems:
Attempt to solve the sequence problem, but heavily over-focus on computationally intensive routing optimization
        │
        ▼
Research Gap:
Lack of direct empirical comparison between CBF and CF under identical experimental conditions for sequential multi-destination recommendation
        │
        ▼
Current Study:
Develops both algorithms within a shared two-stage framework, applying a simple constraint-based approach to isolate and compare preference-filtering performance


## 2.1.1 Notes (Foundations of Travel Recommender Systems)

Key papers:
- Borràs, J., Moreno, A., & Valls, A. (2014). Intelligent tourism recommender systems: A survey.
- Wang, Z., Höpken, W., & Jannach, D. (2023). A survey on point-of-interest recommendations leveraging heterogeneous data.
- Pereira, R. S., Di Sipio, C., De Sanctis, M., & Iovino, L. (2024). On the need for configurable travel recommender systems.

Main ideas:
- **Definition & Purpose:** Travel Recommender Systems (TRS) are a specialized class of recommender systems designed to alleviate cognitive fatigue and information overload by automatically parsing massive datasets to generate shorter, personalized lists of destination suggestions.
- **Domain Differences:** Tourism recommendation fundamentally differs from standard high-frequency e-commerce (e.g., Netflix, Amazon) due to higher financial costs, higher emotional investment, and lower opportunities to undo a poor choice.
- **Data Sparsity:** Low consumption frequency (users only travel a few times a year) creates extreme data sparsity in tourist-attraction interaction matrices.
- **Evolutionary Shift:** The domain has evolved from static information retrieval platforms (electronic directories) into proactive, data-driven intelligent systems that leverage heterogeneous data streams to estimate user satisfaction rather than just presenting raw data.


## 2.1.2 Notes (Traditional Recommendation Algorithms in Tourism)

Key papers:
- García-Crespo, Á., Sánchez-Figueroa, F., & Gutiérrez-López, J. (2020). Content-based recommendations for tourism using linked data.
- Tewari, A. S., & Barman, A. G. (2020). Collaborative filtering based travel recommendation system utilizing context information.
- Goel, S., & Rizvi, S. W. A. (2024). Travel recommendation system using content and collaborative filtering.

CBF (Content-Based Filtering):
- **Mechanism:** Compares multi-dimensional attraction attributes (categories, features) with user profile preference vectors using similarity scoring.
- **Advantages:** Completely immune to the item cold-start problem; newly added attractions can be recommended immediately as long as their descriptive tags are available.
- **Limitations:** Suffers from over-specialization, repeatedly recommending highly homogeneous attractions and limiting discovery.

CF (Collaborative Filtering):
- **Mechanism:** Leverages collective intelligence and community behavioral trends by identifying patterns in user-item rating histories or interaction matrices.
- **Advantages:** Capable of capturing hidden preference patterns and delivering serendipitous discoveries outside the user's explicit profile tags.
- **Limitations:** Heavily vulnerable to data sparsity and the user cold-start problem; neighborhood similarity calculations break down when handling users with zero or limited interaction history.

Comparison & Baseline Justification:
- Goel and Rizvi (2024) demonstrated that the performance of both filters varies based on matrix density and user interaction conditions.
- Despite the rise of deep learning, CBF and CF remain vital baseline choices for this project due to their computational efficiency, lack of massive hardware requirements, and high explainability, making them highly appropriate for a controlled undergraduate comparative experiment.


## 2.1.3 Notes (Multi-Destination Recommendation and Itinerary Generation)

Key papers:
- Lim, K. H., Chan, J., Karunasekera, S., & Leckie, C. (2019). Tour recommendation and itinerary planning: A survey.
- Chen, Y. L., & Cheng, T. H. (2020). A personalized itinerary recommendation system based on user preferences and constraints.
- Otaki, K., & Baba, Y. (2024). Travel itinerary recommendation using interaction-based augmented data.

Main ideas:
- **Flat List Limitation:** Traditional point-wise POI recommendation engines output unordered lists that do not match sequential real-world travel behavior. This forces users to manually figure out combinations, shifting the planning burden back onto the traveler.
- **Two-Stage Architecture Matrix:** Bounded by Lim et al. (2019), effective systems decouple the project into: (Stage 1) Upstream preference estimation/filtering layer, and (Stage 2) Downstream sequence construction/presentation layer.
- **Critique of Route Optimization:** Earlier works treat itinerary generation as an operations-research routing problem (e.g., TSP or Orienteering problem). While logistically efficient, they prioritize path minimization over user preference matching and personalization.
- **Constraint-Based Sequencing:** Chen & Cheng (2020) and Otaki & Baba (2024) show that applying straightforward constraints (like trip duration and categories) after preference filtering serves as an elegant, lower-complexity alternative that keeps the research focus entirely on recommendation algorithm performance.


## 2.2 Notes (Research Gap and Justification)

Core Research Gaps:
1. **Output Format Imbalance:** Most empirical studies focus on single-item recommendations rather than multi-destination contexts. Quantified by Pereira et al. (2024): 87% of systems output flat lists, while only 13% support complete travel planning.
2. **Methodological Disconnect:** Itinerary-related studies overwhelmingly focus on computationally intensive route optimization math rather than evaluating the accuracy of the underlying preference filters.
3. **Lack of Controlled Benchmarking:** Limited research directly compares CBF and CF under identical experimental conditions (same dataset, same metrics) within a shared multi-destination sequence formatting framework.

Project Justification:
- This study directly addresses these gaps by developing both engines within a uniform evaluation framework using a dataset of 100,000 interactions (10,000 tourists, 431 attractions).
- By separating the recommendation filtering from sequence layout via a simple constraint-based presentation layer, the project ensures a fair, scientifically rigorous comparison of CBF and CF.
- The findings will provide clean, actionable baseline references for future researchers and developers selecting appropriate recommendation techniques for multi-destination frameworks.