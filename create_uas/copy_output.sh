gas=$1
if [[ $# -eq 0 ]] ; then
	echo "Provide gas as command line argument (ch4 or co2)."
    exit 1
fi

cp -v SYNTH_SPECTRA/ATM_000001.nc SYNTH_SPECTRA/ATM_${gas}.nc
cp -v SYNTH_SPECTRA/L1B_000001.nc SYNTH_SPECTRA/L1B_${gas}.nc
