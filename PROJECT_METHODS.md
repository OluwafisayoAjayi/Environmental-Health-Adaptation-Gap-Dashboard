# PhD-Standard Project Design

## Working title
**Environmental Health Adaptation Gaps Across U.S. Counties**

## Main question
Which U.S. counties face the greatest gap between environmental-health risk and local adaptive capacity?

## Motivation
Environmental risk does not translate into equal public-health consequences everywhere. Counties with similar climate or pollution burdens may differ in health vulnerability, social vulnerability, infrastructure access, insurance coverage, transportation access, and community resilience. This project develops a county-level framework for identifying places where environmental-health risks are high but adaptive capacity is weak.

## Conceptual framework
The project separates four linked dimensions:

1. **Climate/Natural Hazard Burden**: expected risk from heat, wildfire, flood, drought, storms, and other hazards.
2. **Air Pollution Exposure**: PM2.5, ozone, AQI, and unhealthy air days.
3. **Health Vulnerability**: asthma, COPD, heart disease, poor physical health, poor mental health, and depression.
4. **Adaptive Capacity Deficit**: social vulnerability, poverty, uninsured rate, limited vehicle access, limited internet access, renter share, older population, and weak community resilience.

## Index
The dashboard constructs the Environmental Health Adaptation Gap Index:

EHAGI = 0.30(Climate/Natural Hazard Burden)
      + 0.25(Air Pollution Exposure)
      + 0.25(Health Vulnerability)
      + 0.20(Adaptive Capacity Deficit)

All variables are converted to national percentile ranks before aggregation. Higher values indicate higher risk, vulnerability, or deficit.

## Descriptive model
A first paper can estimate:

Y_c = alpha + beta EHAGI_c + X'_c gamma + delta_s + epsilon_c

where Y_c may represent public-health vulnerability, hazard burden, or adaptation deficits, and delta_s are state fixed effects. This is descriptive and should not be interpreted causally.

## Causal extension
A later dissertation chapter could use an event-study or difference-in-differences design around climate adaptation funding, extreme weather events, pollution regulation, wildfire smoke shocks, or infrastructure investments.

## Policy contribution
The dashboard provides a policy-priority tool that helps identify counties where public-health preparedness, climate adaptation, environmental monitoring, and infrastructure investment may be especially important.

## Limitations
1. The index is not causal.
2. County averages can hide within-county inequality.
3. Some health measures from CDC PLACES are modeled estimates.
4. Air pollution monitor coverage is incomplete in some counties.
5. Weighting choices should be tested in robustness checks.
