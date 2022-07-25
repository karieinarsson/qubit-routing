import json
import sys
import getopt
import Train

def main(argv):
   json_file = ''
   outputfile = ''
   try:
       opts, args = getopt.getopt(argv,"h:j:o:",["jsonfile=","ofile="])
   except getopt.GetoptError:
       print('test.py -i <inputfile> -o <outputfile>')
       sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-j", "--jsonfile"):
         json_file = arg 
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print ('Json file is ' + json_file)
   #print ('Output file is ' + outputfile)
   read_json(json_file)

def read_json(json_file): 
    with open(json_file) as f:
        json_data = json.load(f)

    #print(type(json_data["batch_list"]),json_data["batch_list"])

    
    for item in json_data["batch_list"]:
        print("New training")
        Train.Training(
        depth = item["depth"],
        rows = item["rows"],
        cols = item["cols"],
        max_swaps_per_time_step = item["max_swaps_per_time_step"],
        n_envs = item["n_envs"],
        learning_starts = item["learning_starts"],
        verbose = item["verbose"],
        exploration_fraction = item["exploration_fraction"],
        exploration_initial_eps = item["exploration_initial_eps"],
        exploration_final_eps = item["exploration_final_eps"],
        batch_size = item["batch_size"],
        learning_rate = item["learning_rate"],
        target_update_interval = item["target_update_interval"],
        tau = item["tau"],
        gamma = item["gamma"],
        train_freq = item["train_freq"],
        total_timesteps = item["total_timesteps"],
        log_interval = item["log_interval"],
        eval_freq = item["eval_freq"],
        n_eval_episodes = item["n_eval_episodes"],
        model_dir_add = item["model_dir_add"],
        logdir_add = item["logdir_add"])


if __name__ == "__main__":
    main(sys.argv[1:])

# if __name__ == "__main__":
#    if len(sys.argv) == 1:
#        print("need atleast one argumnet")
#
#    print(f"Arguments count: {len(sys.argv)}")
#    for i, arg in enumerate(sys.argv):
#        print(f"Argument {i:>6}: {arg}")
