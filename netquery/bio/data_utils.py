import netquery.data_utils as utils

# Make training data for Bio
utils.make_train_test_edge_data("/dfs/scratch0/nqe-bio/")
utils.make_train_test_query_data("/dfs/scratch0/nqe-bio/")
utils.sample_new_clean("/dfs/scratch0/nqe-bio/")
utils.clean_test("/dfs/scratch0/nqe-bio/")

