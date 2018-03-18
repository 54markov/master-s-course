cd sdot
make
echo ""
echo "######################### SDOT ###########################"
echo ""
./sdot_sse
./sdot_avx
echo ""
echo "##########################################################"
echo ""
make clean
cd ..

cd loop-if
make
echo ""
echo "######################### LOOP-IF #########################"
echo ""
./sqrt_sse
./sqrt_avx
echo ""
echo "##########################################################"
echo ""
make clean
cd ..