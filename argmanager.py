import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="file",
                        help="Maybe use file or camera")
    
    parser.add_argument("-t",
                        "--type",
                        type=str,
                        default="bot",
                        help="DRAM is top or bot")

    parser.add_argument("-i",
                        "--iVit",
                        action="store_true",
                        help="if model from iVit ")

    parser.add_argument("-p",
                        "--srcpath",
                        type=str,
                        default=os.path.join(os.getcwd(),f"src"),
                        help="srcpath need folder or file")

    parser.add_argument("-s",
                        "--dstpath",
                        type=str,
                        default=os.path.join(os.getcwd(),f"dst"),
                        help="dstpath need folder or file")

    parser.add_argument("-c",
                        "--configpath",
                        type=str,
                        default=os.path.join(os.getcwd(),f"config.ini"),
                        help="configpath need file")

    args = parser.parse_args()
    if args.type!="top" and args.type!="bot":
        raise Exception("DRAM mode error")
    if args.mode=='file':
        if not os.path.exists(args.srcpath):
            raise Exception("srcpath not find")
    config_dirname =os.path.dirname(args.configpath)
    config_basename = os.path.basename(args.configpath)
    config_filename = os.path.splitext(config_basename)[0]
    if os.path.isfile(os.path.join(config_dirname,f"{config_filename}.ini")):
        args.configpath = os.path.join(config_dirname,f"{config_filename}.ini")
    elif os.path.isfile(os.path.join(config_dirname,f"{config_filename}.INI")):
        args.configpath = os.path.join(config_dirname,f"{config_filename}.ini")
    else:    
        raise Exception("config.ini not find")
    return args