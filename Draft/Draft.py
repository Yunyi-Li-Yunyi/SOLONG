
valid1 = adni[(np.isnan(adni["WHOLECEREBELLUM_SUVR"]) == 0)
                   & (np.isnan(adni["YearsOnsetAv45"]) == 0)
                   & (adni["YearsOnsetAv45"] >= 0)]