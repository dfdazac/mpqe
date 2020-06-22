import mpqe.data_utils as utils

# Make training data for Bio
utils.make_train_test_edge_data("/dfs/scratch0/nqe-bio/")
utils.make_train_queries("/dfs/scratch0/nqe-bio/")
utils.make_test_queries("/dfs/scratch0/nqe-bio/")
utils.clean_test_queries("/dfs/scratch0/nqe-bio/")

