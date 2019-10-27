## File Organization
```
/data/embedding -> embedding output
/data/raw -> raw data
/data/clustering -> clustering output
/embedding -> embedding code
/clustering -> clustering code
```

## Pipeline
```
articles.txt --[read_article]--> articles_df ------|               
                                                   |-> embedings --[calculate_distance]--> distance_matrix 
fingerprints.txt-[read_article]-> fingerprints_df -|                   (similarity)                | 
                                                                                                   |
                                                             Evaluation <-- clusters <-- [clustering_algorithms]           
```
