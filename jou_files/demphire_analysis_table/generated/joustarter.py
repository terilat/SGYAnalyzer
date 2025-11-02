import os
import os.path
import optparse

from fnmatch import fnmatch

if __name__ == '__main__':
    usage = "usage: %prog --dir \"workdir\" --fdir \"fidesys.exe path\""
    version="%prog 1.0"
    current_dir = os.path.dirname(__file__)
    fidesys_path = "C:\\Program Files\\Fidesys\\CAE-Fidesys-8.1\\preprocessor\\bin"
    
    parser = optparse.OptionParser(usage=usage, version=version)
    parser.add_option("-d", "--dir", dest="directory", 
                                     help="Kickstarter working directory",
                                     default = current_dir) 
    parser.add_option("-f", "--fdir", dest="fidesys_directory", 
                                     help="Path to fidesys.exe",
                                     default = fidesys_path) 
    
    opts, args = parser.parse_args()
    path = opts.directory
    
    if os.path.isdir(opts.directory) is False:
        raise IOError("Working dirrectory doesn't exist")
    
    if os.path.isdir(opts.fidesys_directory) is False:
        raise IOError("Wrong path to fidesys.exe")

    fidesys_path = os.path.join(opts.fidesys_directory, "fidesysl.exe")#.com")

    print('PATH:', path)
    
    pattern = "*.jou"
    for r,d,f in os.walk(path):
        for file in f:            
            if fnmatch(file, pattern):
                full_path = os.path.join(r, file)
                
                print("\nrun %s" % full_path)
                os.chdir(r)
                os.system("\"" + fidesys_path + "\"" + " -nojournal -nographics -workingdir " + r + " -batch " + full_path)
                print("\"" + fidesys_path + "\"" + " -nojournal -nographics -workingdir " + r + " -batch " + full_path)
                # os.system("\"" + fidesys_path + "\"" + " -nojournal -nographics -workingdir " + r + " -batch " + file )
                
    print("\nAll journal files were launched.")

