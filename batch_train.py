'''
To execute batch training
'''
import json
import sys
import getopt
from datetime import date
import jsonschema
from train import train


class BatchTrain:
    '''
    A class to make batch training possible
    '''
    def __init__(self, json_file: str, schema_file: str):
        """
        :param: json_file: The file path to the json file with the batch trining information (example /JSON_files/basetest.json)
        :param: schema_file : The file path to the schema file to check the json file against (example /JSON_files/schema.json)
        """
        self.json_file = json_file
        self.schema_file = schema_file

    def validate_json(self) -> tuple:
        """
        :return: Boolean to tell if the json file validates against the schema
        :return: A message
        """ 
        json_data = self.__read_json()
        json_schema = self.__get_schema()
        try:
            jsonschema.validate(instance=json_data, schema=json_schema)
        except jsonschema.exceptions.ValidationError as err:
            print(err)
            err = "Given JSON data is InValid"
            return False, err

        message = "Given JSON data is Valid"
        return True, message

    def __get_schema(self) -> dict:
        """
        :return: The schema in a readable format
        """ 
        with open(self.schema_file, 'r', encoding='utf-8') as file:
            schema = json.load(file)
        return schema

    def __read_json(self) -> dict:
        """
        :return: The json_data in a readable format
        """
        with open(self.json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        return json_data

    def do_training(self):
        """
        Uses the data from json_file to do training by calling train 
        and saves it to a file based on the date
        """
        json_data = self.__read_json()
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
            train(
                depth=item["depth"],
                rows=item["rows"],
                cols=item["cols"],
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

            return


def main(argv: list[str]):
    """
    Main function takes in arguments from file call and executes batch training 

    :param: The arguments given when program is started
    """

    json_file = ''
    schema_file = ''
    try:
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

    batch_class = BatchTrain(json_file, schema_file)

    boolean_value, error_or_message = batch_class.validate_json()
    if boolean_value:
        print(error_or_message)
        batch_class.do_training()
    else:
        print(error_or_message)
    
    return

if __name__ == "__main__":
    main(sys.argv[1:])
