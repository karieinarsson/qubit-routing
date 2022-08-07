'''
To execute batch training
'''
import json
import sys
import getopt
from datetime import date
import jsonschema
from train import training


def __validate_json(json_data, json_schema):
    try:
        jsonschema.validate(instance=json_data, schema=json_schema)
    except jsonschema.exceptions.ValidationError as err:
        print(err)
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message


def __get_schema(schema_file):
    with open(schema_file, 'r', encoding='utf-8') as file:
        schema = json.load(file)
    return schema


def __read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data_from_file = json.load(file)
    return json_data_from_file

    # print(type(json_data["batch_list"]),json_data["batch_list"])


def __do_training(json_data):
    for item in json_data["batch_list"]:
        # add date to directory names
        if item["logdir_add"] != "":
            logdir_add = item["logdir_add"] + \
                "-" + str(date.today()) + "/"
        else:
            logdir_add = item["logdir_add"]

        if item["model_dir_add"] != "":
            model_dir_add = item["model_dir_add"] + \
                "-" + str(date.today()) + "/"
        else:
            model_dir_add = item["model_dir_add"]

        print("New training begins")
        training(
            depth=item["depth"],
            rows=item["rows"],
            cols=item["cols"],
            max_swaps_per_time_step=item["max_swaps_per_time_step"],
            n_envs=item["n_envs"],
            learning_starts=item["learning_starts"],
            verbose=item["verbose"],
            exploration_fraction=item["exploration_fraction"],
            exploration_initial_eps=item["exploration_initial_eps"],
            exploration_final_eps=item["exploration_final_eps"],
            batch_size=item["batch_size"],
            learning_rate=item["learning_rate"],
            target_update_interval=item["target_update_interval"],
            tau=item["tau"],
            gamma=item["gamma"],
            train_freq=item["train_freq"],
            total_timesteps=item["total_timesteps"],
            log_interval=item["log_interval"],
            eval_freq=item["eval_freq"],
            n_eval_episodes=item["n_eval_episodes"],
            model_dir_add=model_dir_add,
            logdir_add=logdir_add)


def main(argv):
    '''main function takes in arguments from file call and executes batch training '''
    json_file = ''
    schema_file = ''
    try:
        # add ,args after opts to get arguments without -LETTER
        opts, args = getopt.getopt(
            argv, "h:j:s:", ["jsonfile=", "schemafile="])
    except getopt.GetoptError:
        print('test.py -j <jsonfile> -s <schemafile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('PYTHONFILE.py -j <jsonfile> -s <schemafile>')
            sys.exit()
        elif opt in ("-j", "--jsonfile"):
            json_file = arg
        elif opt in ("-s", "--schemafile"):
            schema_file = arg
    print('Json file is ' + json_file)
    print('Schema file is ' + schema_file)

    json_data = __read_json(json_file)
    json_schema = __get_schema(schema_file)
    boolean_value, error_or_message = __validate_json(json_data, json_schema)
    if boolean_value:
        print(error_or_message)
        __do_training(json_data)
    else:
        print(error_or_message)


if __name__ == "__main__":
    main(sys.argv[1:])

# if __name__ == "__main__":
#    if len(sys.argv) == 1:
#        print("need atleast one argumnet")
#
#    print(f"Arguments count: {len(sys.argv)}")
#    for i, arg in enumerate(sys.argv):
#        print(f"Argument {i:>6}: {arg}")
