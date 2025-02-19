gas=$1
if [[ $# -eq 0 ]] ; then
	echo "Provide gas as command line argument (ch4 or co2)."
    exit 1
fi

cp -v INPUT/${gas}.nc INPUT/RTC_INP_DATA.nc
cp -v INI/settings_RTC_create_${gas}.nml INI/settings_RTC_create_000001.nml
cp -v INI/syn_create.nml INI/syn_create_000001.nml
