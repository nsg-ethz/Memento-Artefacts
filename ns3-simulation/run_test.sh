#! /bin/bash
waf build
waf --run-no-build "trafficgen --prefix=50 --congestion=0Mbps"
waf --run-no-build "trafficgen --prefix=70 --congestion=20Mbps"
waf --run-no-build "trafficgen --prefix=90 --congestion=40Mbps"
#waf --run-no-build "trafficgen --prefix=150 --congestion=100Mbps"
