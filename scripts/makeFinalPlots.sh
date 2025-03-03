#!/usr/bin/env bash

### Defaults
DRYRUN="0"
NOSIG="0"
NODATA="0"
TAG=""
CHANNEL="ETau"
CHANNEL_CHOICES=("ETau" "MuTau" "TauTau")
SELECTION="baseline"
SELECTION_CHOICES=( "baseline" "s1b1jresolvedMcut" "s2b0jresolvedMcut" "sboostedLLMcut" )
DATA_PERIOD="UL18"
DATA_PERIOD_CHOICES=( "UL16" "UL17" "UL18" )
REG="SR"  # A:SR , B:SStight , C:OSinviso, D:SSinviso, B': SSrlx
REG_CHOICES=( "SR" "SStight" "OSinviso" "SSinviso" )
EOS_USER="bfontana"
CFGFILE="mainCfg_${CHANNEL}_${DATA_PERIOD}.cfg"

### Argument parsing
HELP_STR="Prints this help message."
CHANNEL_STR="(String) Which channel to consider: ${CHANNEL_CHOICES[@]}. Defaults to '${CHANNEL}'."
SELECTION_STR="(String) Which selection to consider: ${SELECTION_CHOICES[@]}. Defaults to '${SELECTION}'."
DRYRUN_STR="(Boolean) Prints all the commands to be launched but does not launch them. Defaults to ${DRYRUN}."
TAG_STR="(String) Defines tag for the output. Defaults to '${TAG}'."
DATAPERIOD_STR="(String) Which data period to consider: Legacy18, UL18, ... Defaults to '${DATA_PERIOD}'."
REG_STR="(String) Which region to consider: A: SR, B: SStight, C: OSinviso, D: SSinviso, B': SSrlx. Defaults to '${REG}'."
NOSIG_STR="(Boolean) Do not include signal samples. Defaults to '${NOSIG}'."
NODATA_STR="(Boolean) Do not include data samples. Defaults to '${NODATA}'."
CFGFILE_STR="(String) Configuration file used to retrieve some information. Defaults to '${CFG_FILE}'."
function print_usage_submit_skims {
    USAGE=" 
    Run example: bash $(basename "$0") -t <some_tag>

    -h / --help		    [${HELP_STR}]
    --dryrun		    [${DRYRUN_STR}]
    -c / --channel      [${CHANNEL_STR}]
    -s / --selection    [${SELECTION_STR}]
    -t / --tag		    [${TAG_STR}]
    -r / --region    	[${REG_STR}]
    -d / --data_period  [${DATAPERIOD_STR}]
    --nosig             [${NOSIG_STR}]
    --nodata            [${NODATA_STR}]
    --cfg               [${CFGFILE_STR}]

"
    printf "${USAGE}"
}

while [[ $# -gt 0 ]]; do
    key=${1}
    case $key in
	-h|--help)
	    print_usage_submit_skims
	    exit 1
	    ;;
	-c|--channel)
		CHANNEL=${2}
		if [[ ! " ${CHANNEL_CHOICES[*]} " =~ " ${CHANNEL} " ]]; then
			echo "Currently the following channels are supported:"
			for ch in ${CHANNEL_CHOICES[@]}; do
				echo "- ${ch}" # bash string substitution
			done
			exit 1;
		fi
		shift; shift;
		;;
	-s|--selection)
		SELECTION=${2}
		if [[ ! " ${SELECTION_CHOICES[*]} " =~ " ${SELECTION} " ]]; then
			echo "Currently the following selections are supported:"
			for sl in ${SELECTION_CHOICES[@]}; do
				echo "- ${sl}" # bash string substitution
			done
			exit 1;
		fi
		shift; shift;
		;;
	--dryrun)
	    DRYRUN="1"
	    shift;
	    ;;
	--nosig)
	    NOSIG="1"
	    shift;
	    ;;
	--nodata)
	    NODATA="1"
	    shift;
	    ;;
	-t|--tag)
	    TAG=${2}
	    shift; shift;
	    ;;
	-d|--data_period)
	    DATA_PERIOD=${2}
		if [[ ! " ${DATA_PERIOD_CHOICES[*]} " =~ " ${DATA_PERIOD} " ]]; then
			echo "Currently the following data periods are supported:"
			for dp in ${DATA_PERIOD_CHOICES[@]}; do
				echo "- ${dp}" # bash string substitution
			done
			exit 1;
		fi
	    shift; shift;
	    ;;
	--eos)
	    EOS_USER="$2"
	    shift # past argument
	    shift # past value
	    ;;
	--cfg)
	    CFGFILE="$2"
	    shift # past argument
	    shift # past value
	    ;;
	-r|--region)
	    REG=${2}
		if [[ ! " ${REG_CHOICES[*]} " =~ " ${REG} " ]]; then
			echo "Currently the following regions are supported:"
			for rg in ${REG_CHOICES[@]}; do
				echo "- ${rg}" # bash string substitution
			done
			exit 1;
		fi
	    shift; shift;
	    ;;
	*)  # unknown option
	    echo "Wrong parameter ${1}."
	    exit 1
	    ;;
    esac
done

### Setup variables
PLOTS_DIR="HH_Plots"
MAIN_DIR="/data_CMS/cms/${USER}/HHresonant_hist"

### Argument parsing sanity checks
if [[ -z ${TAG} ]]; then
    printf "Select the tag via the '--tag' option. "
    declare -a tags=( $(/bin/ls -d ${MAIN_DIR}/*/ | tr '\n' '\0' | xargs -0 -n 1 basename) )
    if [ ${#tags[@]} -ne 0 ]; then
		echo "The following tags are currently available:"
		for tag in ${tags[@]}; do
			echo "- ${tag}"
		done
    else
		echo "No tags are currently available. Everything looks clean!"
    fi
    exit 1;
fi
if [[ -z ${DATA_PERIOD} ]]; then
	echo "Select the data period via the '-d / --data_period' option."
	exit 1;
fi
if [[ -z ${CHANNEL} ]]; then
	echo "Select the channel via the '-c / --channel' option."
	exit 1;
fi
if [[ -z ${REG} ]]; then
	echo "Select the region via the '-r / --region' option."
	exit 1;
fi

LUMI="59.9"
MAIN_DIR="${MAIN_DIR}/${TAG}"
EOS_DIR="/eos/user/${EOS_USER:0:1}/${EOS_USER}"
WWW_DIR="${EOS_DIR}/www/${PLOTS_DIR}/${TAG}/${CHANNEL}"
WWW_SUBDIR="${WWW_DIR}/${SELECTION}_${REG}"

[[ ! -d ${EOS_DIR} ]] && /opt/exp_soft/cms/t3/eos-login -username ${EOS_USER} -init

if [ ${DATA_PERIOD} == "UL16" ]; then
	PLOTTER="scripts/makeFinalPlots_UL2016.py"
elif [ ${DATA_PERIOD} == "UL17" ]; then
	PLOTTER="scripts/makeFinalPlots_UL2017.py"
elif [ ${DATA_PERIOD} == "UL18" ]; then
	PLOTTER="scripts/makeFinalPlots_UL2018.py"
fi

### Argument parsing: information for the user
echo "------ Arguments --------------"
printf "DRYRUN\t\t= ${DRYRUN}\n"
printf "TAG\t\t= ${TAG}\n"
printf "DATA_PERIOD\t= ${DATA_PERIOD}\n"
printf "CHANNEL\t\t= ${CHANNEL}\n"
printf "SELECTION\t= ${SELECTION}\n"
printf "REGION\t\t= ${REG}\n"
printf "EOS_USER\t= ${EOS_USER}\n"
printf "NOSIG\t\t= ${NOSIG}\n"
printf "NODATA\t\t= ${NODATA}\n"
printf "CFGFILE\t\t= ${CFGFILE}\n"
echo "-------------------------------"

### Ensure connection to /eos/ folder
[[ ! -d ${EOS_DIR} ]] && /opt/exp_soft/cms/t3/eos-login -username ${EOS_USER} -init

OPTIONS="--quit --ratio " #"--binwidth"
if [[ ${NOSIG} -eq 0 ]]; then
    OPTIONS+=" --signals ggFRadion280 ggFRadion400 ggFRadion550 ggFRadion800 ggFRadion1500 "
else
    OPTIONS+=" --nosig "
fi

if [[ ${NODATA} -eq 0 ]]; then
    OPTIONS+=" --nodata "
fi

OUTDIR="${MAIN_DIR}/${PLOTS_DIR}"
FULL_OUTDIR="${MAIN_DIR}/${PLOTS_DIR}/${CHANNEL}/${SELECTION}_${REG}"
mkdir -p "${FULL_OUTDIR}"

function run() {
	[[ ${DRYRUN} -eq 1 ]] && echo "[DRYRUN] $@" || "$@"
}

function run_plot() {
	comm="python ${PLOTTER} --indir ${MAIN_DIR} --outdir ${OUTDIR} "
	comm+="--reg ${REG} "
	comm+="--sel ${SELECTION} --channel ${CHANNEL} "
	comm+="--cfg ${CFGFILE} "
	comm+="--lumi ${LUMI} ${OPTIONS} $@"
	[[ ${DRYRUN} -eq 1 ]] && echo "[DRYRUN] ${comm}" || ${comm}
}

declare -A VAR_MAP
VAR_MAP=(
	["dau1_pt"]="pT_{1}[GeV] "
	["bjet1_pt"]="pT_{j1}[GeV] "
	["bjet2_pt"]="pT_{j2}[GeV] "
	["dau1_eta"]="eta_{1}[GeV] "
	["bjet1_eta"]="eta_{j1}[GeV] "
	["bjet2_eta"]="eta_{j2}[GeV] "
	["tauH_mass"]="m_{H#tau}[Gev] "
	["tauH_mass"]="m_{H#tau}[Gev] --equalwidth "
	["tauH_pt"]="pT_{H#tau}[Gev] "
	["tauH_SVFIT_mass"]="m_{H#tau_{SVFit}}[Gev] "
	["bH_mass"]="m_{Hb}[Gev] "
	["bH_pt"]="pT_{Hb}[Gev]	"
	["ditau_deltaR"]="#DeltaR(#tau#tau)	"
	["dib_deltaR"]="#DeltaR(bb)	"
	["HH_deltaR"]="#DeltaR(HH) "
	["HH_mass"]="m_{HH}[GeV] --logy "
	["HHKin_mass"]="m_{HHKin}[GeV] --logy "
	["HH_mass"]="m_{HH}[GeV] --logy --logx "
	["HHKin_mass"]="m_{HHKin}[GeV] --logy --logx "
	["DNNoutSM_kl_1"]="DNN --logy --equalwidth "
)
declare -A SIGSCALE_CHANNEL_MAP
SIGSCALE_CHANNEL_MAP=(
	["dau1_pt"]="100 50 20"
	["bjet1_pt"]="100 100 20 "
	["bjet2_pt"]="100 100 20 "
	["dau1_eta"]="40 40 2 "
	["bjet1_eta"]="40 40 4 "
	["bjet2_eta"]="40 40 4 "
	["tauH_mass"]="30 30 2 "
	["tauH_pt"]="30 30 10 "
	["tauH_SVFIT_mass"]="10 10 2 "
	["bH_mass"]="30 30 2 "
	["bH_pt"]="100 100 5"
	["ditau_deltaR"]="10 10 1 "
	["dib_deltaR"]="10 10 1 "
	["HH_deltaR"]="20 20 2 "
	["HH_mass"]="5 5 1 "
	["HHKin_mass"]="5 5 1 "
	["HH_mass"]="5 5 1 "
	["HHKin_mass"]="5 5 1 "
	["DNNoutSM_kl_1"]="5 5 1 "
)

for avar in ${!VAR_MAP[@]}; do
	ssarr=(${SIGSCALE_CHANNEL_MAP[${avar}]})
	if [ ${CHANNEL} == "ETau" ]; then
		ss=${ssarr[0]}
	elif [ ${CHANNEL} == "MuTau" ]; then
		ss=${ssarr[1]}
	elif [ ${CHANNEL} == "TauTau" ]; then
		ss=${ssarr[2]}
	else
		echo "Channel ${CHANNEL} is not supported." 
	fi
    run_plot --var ${avar} --lymin 0.7 --sigscale ${ss} --label ${VAR_MAP[$avar]}
done

run mkdir -p ${WWW_DIR}
if [ -d ${WWW_SUBDIR} ]; then
    run rm -rf ${WWW_SUBDIR}
fi

run mkdir ${WWW_SUBDIR}

run cp ${FULL_OUTDIR}/*png ${WWW_SUBDIR}
run cp ${FULL_OUTDIR}/*pdf ${WWW_SUBDIR}

echo "Results: https://${EOS_USER}.web.cern.ch/${EOS_USER}/${PLOTS_DIR}/${TAG}/${CHANNEL}/${SELECTION}_${REG}/"
