#!/bin/sh
#
# A bash script to create a virtual envrionment and install the python packages.
#
# copyright 2020 Artelsys <contact@artelsys.com> (http://www.artelsys.com/)
#
#
# Redistribution and use of this script, with or without modification, is
# permitted provided that the following conditions are met:
#
# 1. Redistributions of this script must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
#  EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

VENV=".venv"


# --- Private Functions --------------------------------------------------------

##
# Dumps the help message.
#
function help() {
  echo ''
    echo 'Usage: -v <virtual environment>'
    echo ''
    echo '-h  print the help function,'
    echo "-v  the name of the virtual environment. Default is ${VENV},"
    echo '-f  force to replace the existing virtual environment,'
    echo ''
}


# --- Main program -------------------------------------------------------------

venv=${VENV}

# Collect the passed in options:
while getopts "hv:f" opt; do
  case ${opt} in

    h ) help
        exit 1
      ;;

    v ) venv=$(echo ${OPTARG} | tr '[:upper:]' '[:lower:]')
        ;;

    f ) FORCE=true
        ;;

    : ) echo "Option -$OPTARG requires an argument" >&2
        help
        exit 1
        ;;

  esac
done


# Check if the virtual environment already exist?
if [[ -d ${venv} ]]
then
  if [[ ${FORCE} = true ]]
    then
      rm -rf ${venv}
    else
      echo "the virtual environment ${venv} already exists! Use the option -f to replace it."
      exit 1
  fi
fi


# Create the virtual environment
echo "Creating the virtual environment ${venv} ...:"
python3 -m venv ./${venv}
echo '  done.'

# Activate the environment
echo 'Activating the environment ...:'
source ./${venv}/bin/activate
echo '  done.'

# Update pip and setuptools
echo 'Updating pip and setuptools ...:'
pip install --upgrade pip setuptools
echo '  done.'
echo ''

# Install the packages required by Atom-IDE
echo 'Installing the packages required by Atom ...:'
source ./scripts/python_atom_packages_install.sh
echo '  done.'
echo ''

# Install the packages required by the project
echo 'Installing the packages required by the project ...:'
source ./scripts/python_project_packages_install.sh
echo '  done.'
echo ''

# reference your virtual environment to `Jupyter`
echo "Adding this virtual environment ${venv} to Jupyter kernels:"
python -m ipykernel install --user --name=${venv}
echo '  done.'

echo 'Done!'

# -- o ---
