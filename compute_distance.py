import numpy as np
import pandas as pd

Pareto_solves = np.array([[5.80500611e-01, 6.89831442e-01, -1.01428529e+00,
                           -1.11296420e+00, 4.28795769e-01, 5.22074259e-01,
                           6.77592327e-01, 8.10727493e-02, -4.54335242e-01,
                           -1.61957026e+00, -1.63158240e+00, -6.93723681e-01,
                           -1.56122616e-01, -1.86521289e+00, 7.76204349e-01,
                           -1.18095006e-01, 1.37500953e+00, -5.34108525e-01,
                           3.74707341e-01, -3.37335108e-01],
                          [-7.80805728e-01, 6.68007245e-01, -1.20002773e+00,
                           -6.26394886e-01, 9.57729151e-02, 1.20269947e+00,
                           3.32295322e-01, 8.15023776e-01, -4.26930645e-01,
                           -6.99714491e-01, -1.01634962e+00, -1.63262410e-01,
                           -1.91944049e-02, -1.46910027e+00, 3.35832035e-01,
                           -2.08339775e+00, 3.15080126e-01, -6.19875384e-01,
                           5.10597753e-01, -1.56152437e+00],
                          [7.17859147e-01, 5.80483123e-01, -6.22285221e-01,
                           -3.46007450e-01, 5.02242619e-01, 1.54815158e+00,
                           2.07675013e-01, 3.55950566e-01, 1.05165719e-01,
                           -1.03126863e+00, -9.88056211e-01, -3.54432341e-01,
                           -2.22551695e-01, -9.24345223e-01, 9.80511044e-01,
                           -2.00145267e+00, 6.74577400e-01, -5.71507293e-01,
                           4.89892837e-02, -6.59495179e-01],
                          [6.85964911e-01, 5.72185381e-01, -5.25974329e-01,
                           -8.96379156e-01, 3.75764112e-01, 1.23613872e+00,
                           -1.50843007e-01, 3.95368590e-01, -1.34727791e+00,
                           -1.42383111e+00, -1.53508698e+00, -6.89410988e-01,
                           -6.35371354e-01, -1.43506849e+00, 8.25516638e-01,
                           -1.40414603e+00, 1.35140355e+00, -4.63360069e-01,
                           3.25535435e-01, -3.70321105e-01],
                          [6.82582978e-01, 6.61073516e-01, -4.70755265e-01,
                           -1.05392071e+00, 4.46260329e-01, 1.45993443e+00,
                           -3.97701581e-02, 4.63249363e-01, -1.11198823e+00,
                           -1.48602215e+00, -1.87036390e+00, -8.61918718e-01,
                           -3.08958072e-02, -1.80158827e+00, 3.95006782e-01,
                           -7.42827824e-02, 2.23515952e-01, -1.89642317e-01,
                           2.52798718e-01, -3.44844888e-01],
                          [-8.01409509e-01, 7.55701869e-01, -9.40577976e-01,
                           -7.38782271e-01, -1.18926786e+00, 1.24817424e+00,
                           -2.03155789e-01, 8.99573341e-01, -6.76954591e-01,
                           -4.96780490e-01, -8.55002722e-01, 9.63617236e-02,
                           -1.84676097e-02, -1.67396910e+00, 4.14550735e-01,
                           -2.01021511e+00, 2.81490997e-01, -5.59875393e-01,
                           2.30595429e-01, -1.32444345e+00],
                          [7.07453197e-01, 4.74772170e-01, -1.54694349e+00,
                           -5.38355334e-01, 5.24491991e-01, 1.34517407e+00,
                           -8.59620996e-02, 3.12129465e-01, -3.27975972e-01,
                           -6.41564389e-01, -1.33733093e+00, -2.87624802e-01,
                           -6.37769779e-01, -1.79289537e+00, 1.94138375e-01,
                           -1.52576551e+00, -1.40244957e-01, -5.40071715e-01,
                           2.18132262e-01, -4.48080603e-01],
                          [7.59326857e-01, 2.10340285e+00, -5.86660018e-01,
                           -1.07467170e+00, 4.42512227e-01, 6.14585161e-01,
                           2.59579739e-01, 4.01344195e-01, -8.67468137e-01,
                           -1.48010858e+00, -1.70090126e+00, -4.57466503e-01,
                           -3.98290768e-01, -2.12223913e+00, 7.92038570e-01,
                           -1.32268774e+00, 7.65829601e-01, -4.03065600e-01,
                           2.61910445e-01, -3.54350939e-01],
                          [-3.89906228e-01, 2.29420459e-01, -5.23640264e-01,
                           -1.10482981e+00, -1.54472706e-01, 9.73438983e-01,
                           -8.29424851e-02, 7.08406408e-01, -3.30555902e-01,
                           -1.18876671e+00, -9.80982858e-01, -3.74976443e-01,
                           1.45860779e-01, -1.53309481e+00, 4.59519924e-01,
                           -1.94441565e+00, 1.64604987e-01, -6.29519555e-01,
                           6.83563468e-01, -1.47330821e+00],
                          [4.37418803e-01, 5.81506132e-01, -7.35364527e-01,
                           -1.01551753e+00, -1.60054986e-01, 4.42249451e-01,
                           2.48643838e-01, 8.36986745e-01, -3.15534973e-01,
                           -5.54043557e-01, -1.09951736e+00, -3.47296794e-01,
                           1.64321377e-01, -1.43882925e+00, 4.82592646e-01,
                           -1.46231892e+00, 7.65309645e-01, -5.76071710e-01,
                           1.50841635e-01, -1.32881624e+00],
                          [4.54276441e-01, 5.82301806e-01, -7.59687941e-01,
                           -1.01546220e+00, 2.39210015e-02, 3.11094755e-01,
                           -1.07017790e-01, 8.07528062e-01, -4.36619718e-01,
                           -5.80063263e-01, -1.09378422e+00, -4.14182746e-01,
                           2.00370418e-01, -1.33124691e+00, 4.53638641e-01,
                           -1.83683230e+00, 2.92878024e-01, -5.99629988e-01,
                           2.64319292e-01, -1.71647300e+00],
                          [4.70561753e-01, 9.85594774e-01, -9.16684522e-01,
                           -8.53881120e-01, -1.11669822e+00, 7.43072524e-01,
                           -3.24348427e-01, 1.19547061e+00, 2.71944816e-02,
                           -5.79866144e-01, -1.08365816e+00, -2.01135699e-01,
                           -1.14767970e-01, -9.33654640e-01, -1.13445762e-02,
                           -1.91423389e+00, 1.45834592e-01, -5.85421402e-01,
                           5.55266162e-01, -1.49916467e+00],
                          [-8.52346632e-01, 9.29329267e-01, -8.08089078e-01,
                           -1.12790492e+00, -1.25769066e+00, 9.41691219e-01,
                           -2.33515157e-01, 1.06295895e+00, -3.77166649e-01,
                           -9.80214816e-01, -1.17404816e+00, -4.04773233e-01,
                           -1.14549932e-01, -1.55140538e+00, 5.40591137e-01,
                           -2.04623850e+00, 3.72379229e-01, -4.98771107e-01,
                           5.45944970e-01, -1.27292066e+00],
                          [4.90905384e-01, 1.18708127e-01, -1.15199513e+00,
                           -5.29335568e-01, 5.26884397e-01, 1.26294216e+00,
                           7.17923581e-02, 4.23254745e-01, 1.77059787e-01,
                           -5.10973057e-01, -1.07628698e+00, -8.43570169e-01,
                           -1.27259278e-02, -1.53161516e+00, 3.68676734e-01,
                           -1.41331414e+00, 5.67414559e-01, -5.87041033e-01,
                           2.08287408e-01, -9.09979625e-01],
                          [-1.00984068e+00, 8.24300320e-01, -1.02902675e+00,
                           -1.06554126e+00, -1.68986633e-01, 9.36486667e-01,
                           -4.21302538e-01, 8.33684437e-01, -3.74472055e-01,
                           -6.39987437e-01, -1.08782771e+00, -3.03542560e-01,
                           -2.40543633e-03, -1.28383670e+00, 6.58397743e-01,
                           -1.75164186e+00, 3.29742873e-01, -4.97666813e-01,
                           8.52758812e-01, -1.33490011e+00],
                          [4.55421096e-01, 7.00800374e-01, -9.40577976e-01,
                           -5.16110268e-01, 2.34133703e-01, 9.19311648e-01,
                           3.73753814e-01, 7.85984434e-01, -4.63622992e-01,
                           -6.59896455e-01, -8.25518009e-01, -1.72044622e-01,
                           -1.86648013e-01, -1.49893973e+00, 5.48010600e-01,
                           -2.08859031e+00, 3.16743984e-01, -5.65396865e-01,
                           4.56974716e-01, -2.09719034e+00],
                          [7.00637300e-01, 2.56757538e-01, -6.26769082e-01,
                           -9.82481949e-01, 4.51922356e-01, 9.04348562e-01,
                           1.09741568e-01, 4.19900020e-01, -5.58277781e-01,
                           -1.49449826e+00, -1.64148510e+00, -7.79193420e-01,
                           -4.10791645e-01, -3.77678972e-01, 1.28796638e+00,
                           -2.06416906e+00, 2.94073922e-01, -6.12807901e-01,
                           1.05597197e-01, -5.76317232e-01],
                          [4.78054036e-01, 4.65735588e-01, -6.07973717e-01,
                           -1.01983374e+00, 4.62927421e-01, 1.34972805e+00,
                           1.53974841e-01, 3.67639687e-01, -3.86913054e-01,
                           -4.78645542e-01, -1.15915689e+00, -5.18157859e-01,
                           -3.04824909e-01, -1.47526545e+00, 6.36410795e-01,
                           -1.96104806e+00, 3.48461273e-01, -5.96611584e-01,
                           1.80009634e-01, -1.77797715e+00],
                          [4.75088341e-01, 4.68349945e-01, -6.48635586e-01,
                           -6.46813864e-01, 4.89243883e-01, 1.25207766e+00,
                           1.81885873e-01, 3.65333314e-01, 4.92212649e-01,
                           -6.80199711e-01, -9.54923137e-01, -6.16173615e-01,
                           -7.35586840e-02, -9.54554590e-01, 8.56099305e-01,
                           -1.45444895e+00, 1.81295561e-01, -5.76734286e-01,
                           3.20455909e-01, -6.42859590e-01],
                          [6.32946598e-01, 9.37115504e-01, -1.73747690e+00,
                           -2.26094371e-01, 4.28875516e-01, 4.73411703e-01,
                           -6.32185890e-01, 3.72462105e-01, -4.00042034e-01,
                           -1.57689400e+00, -1.72539740e+00, -8.90147256e-01,
                           -2.07870432e-01, -1.34844775e+00, 6.83280090e-01,
                           -1.36187831e-01, 1.24143295e+00, -2.34992004e-01,
                           3.83452505e-01, -4.81351782e-01],
                          [5.09219856e-01, 6.18448132e-01, -8.21847776e-01,
                           -1.12779425e+00, -4.84891401e-02, 1.27718962e+00,
                           -1.06936179e-01, 7.52227509e-01, -3.41334279e-01,
                           -5.79274788e-01, -7.79206163e-01, -5.18628335e-01,
                           -1.50235575e-01, -1.29980451e+00, 4.97069648e-01,
                           -1.48227782e+00, 6.49295066e-02, -5.60317111e-01,
                           5.86371713e-01, -1.71780385e+00],
                          [6.49179879e-01, 2.40730394e-01, -5.85615831e-01,
                           -1.40046914e-01, 4.37647669e-01, 1.35226527e+00,
                           3.45761171e-01, 4.24407932e-01, -8.61505630e-01,
                           -1.42185992e+00, -1.88034105e+00, -7.27911577e-01,
                           -9.60166549e-02, -1.85454714e+00, 8.07963273e-01,
                           5.93285686e-01, -2.04471963e-02, -5.30942882e-01,
                           3.33180907e-01, -4.54734839e-01],
                          [4.22017997e-01, 6.76873325e-01, -8.58087206e-01,
                           -6.58157741e-01, 5.33998497e-03, 1.25077652e+00,
                           -7.12720829e-02, 4.90349254e-01, 4.73121162e-01,
                           -9.32709139e-01, -1.01500941e+00, -5.12355327e-01,
                           1.64684774e-01, -1.37008752e+00, 5.07384512e-01,
                           -1.91001494e+00, 3.92969469e-01, -6.36734278e-01,
                           3.19670416e-01, -1.32691503e+00],
                          [5.89917996e-01, 2.24475914e-01, -6.04656888e-01,
                           -8.84205240e-01, 4.51284381e-01, 1.17290342e+00,
                           3.60532799e-01, 1.69136251e-02, -1.38007169e+00,
                           -1.59522607e+00, -1.88987147e+00, -8.82776471e-01,
                           -3.50031569e-01, -8.73050949e-01, 1.28832831e+00,
                           -7.03883625e-02, 8.06854094e-01, -5.23286441e-01,
                           2.12843271e-01, -5.47038595e-01],
                          [5.95068941e-01, 9.04777046e-01, -6.44704529e-01,
                           -1.11086143e+00, 5.29914777e-01, 1.21772762e+00,
                           1.81314594e-01, 3.85514084e-01, -2.10216470e-01,
                           -7.92163298e-01, -1.22892249e+00, -6.22760274e-01,
                           -2.12521921e-01, -9.62322713e-01, 4.77525695e-01,
                           -1.18013872e-01, 1.90030814e-01, -5.69004226e-01,
                           3.73921848e-01, -5.50365712e-01],
                          [-8.08329465e-01, 5.51554695e-01, -1.05070899e+00,
                           -7.27770410e-01, -1.53196757e-01, 9.73373926e-01,
                           -9.86118363e-02, 1.30712007e+00, -3.34110473e-01,
                           -7.39039730e-01, -1.05789626e+00, -3.94422769e-01,
                           -2.72191805e-01, -1.41151751e+00, 8.04344022e-01,
                           -1.94587605e+00, 5.98859376e-02, -5.55458216e-01,
                           7.94056249e-01, -1.33898771e+00],
                          [6.53394289e-01, 8.18730603e-01, -4.90349127e-01,
                           -5.63035183e-01, 7.14608486e-01, 1.45284323e+00,
                           -1.29542483e-01, 4.39818702e-01, -3.93907533e-01,
                           -1.44413436e+00, -1.64066608e+00, -3.93952293e-01,
                           -1.98058698e-01, -9.01657371e-01, 5.00598418e-01,
                           -1.33315400e+00, 7.55690467e-01, -5.22771104e-01,
                           4.35242723e-01, -3.78211127e-01],
                          [4.81279881e-01, 3.50135547e-01, -3.87527422e-01,
                           -1.02304323e+00, -1.69943596e-01, 9.52230435e-01,
                           2.09470460e-01, 4.08158481e-01, -6.23750688e-01,
                           -6.86704638e-01, -1.16623025e+00, -4.37706527e-01,
                           1.60178644e-01, -1.59560970e+00, 1.04801007e+00,
                           -1.95106861e+00, 2.41194441e-01, -6.01323240e-01,
                           1.33927337e-01, -1.56199967e+00],
                          [4.16502844e-01, 7.10245309e-02, -1.12533765e+00,
                           -4.62157682e-01, -6.15676238e-02, 1.41842813e+00,
                           -1.23421642e-01, 4.07424634e-01, 4.05756306e-01,
                           -8.88061688e-01, -1.01083985e+00, -3.47139968e-01,
                           1.48259204e-01, -1.34536516e+00, 5.30366754e-01,
                           -2.24404257e+00, 2.36254863e-01, -5.28366195e-01,
                           3.69680182e-01, -1.21198687e+00],
                          [6.62915733e-01, 7.07904604e-01, -3.72663114e-01,
                           -5.44331620e-01, 5.02242619e-01, 1.41491506e+00,
                           2.02533507e-01, 4.63301780e-01, 9.04887802e-02,
                           -1.01865302e+00, -9.88056211e-01, -3.53177739e-01,
                           -2.22551695e-01, -9.36182363e-01, 6.09899785e-01,
                           -1.33615595e+00, 6.74577400e-01, -6.09200539e-01,
                           5.23407236e-02, -6.59495179e-01],
                          [4.76597203e-01, 6.82329374e-01, -2.03789563e+00,
                           -1.08418949e+00, 4.08221081e-01, 1.24238418e+00,
                           1.01172392e-01, 4.28968262e-01, -9.92221204e-02,
                           -4.54104228e-01, -9.57975848e-01, -6.03549186e-01,
                           2.66290740e-01, -1.42631394e+00, 6.87532710e-01,
                           -1.16147796e-01, 1.52957983e-01, -6.24218942e-01,
                           2.69346452e-01, -5.69282754e-01],
                          [-7.74978396e-01, 7.26375604e-01, -9.13122002e-01,
                           -9.66379177e-01, -1.19963495e+00, 1.22312734e+00,
                           -1.16892746e-01, 1.24752127e+00, -5.61373698e-01,
                           -6.90252780e-01, -7.88364293e-01, 9.76947379e-02,
                           -2.75026307e-01, -1.59992533e+00, 7.02643081e-01,
                           -2.07301263e+00, 3.38062162e-01, -5.45740426e-01,
                           5.92550930e-01, -1.29392903e+00],
                          [7.28265097e-01, 5.47292157e-01, -5.73761238e-01,
                           -1.10720926e+00, 5.03678063e-01, 1.28011718e+00,
                           2.26200757e-01, 3.92852546e-01, -1.25302444e+00,
                           -1.55264837e+00, -1.92062194e+00, -7.63746137e-01,
                           -3.01690120e-02, -6.73792427e-01, 9.30565386e-01,
                           -5.82183005e-02, -1.02080219e-01, 5.45539664e-02,
                           1.05649563e-01, -4.26501867e-01],
                          [6.28576099e-01, 4.44081893e-01, -5.16699492e-01,
                           -6.58102405e-01, 5.03598316e-01, 1.42727587e+00,
                           4.38150344e-03, 2.84662650e-01, -3.95340827e-01,
                           -1.44600699e+00, -1.33896897e+00, -4.26885587e-01,
                           -1.99148890e-01, -1.44154193e+00, 6.26548338e-01,
                           -2.07958447e+00, 9.44109868e-02, -5.24243496e-01,
                           6.62931166e-01, -3.66518685e-01],
                          [7.41324564e-01, 2.20497545e-01, -5.58282702e-01,
                           5.60638167e-02, 4.25366654e-01, 1.25233788e+00,
                           -1.43416387e-01, 4.07319799e-01, -1.32721178e+00,
                           -1.52584018e+00, -1.48698818e+00, -8.50784128e-01,
                           -6.79705860e-01, -1.20979292e+00, -4.02985808e-02,
                           5.04363100e-01, 1.97867794e+00, 1.81326954e-01,
                           2.58925569e-01, -3.00261508e-01],
                          [5.94236465e-01, 5.43086452e-01, -5.74129775e-01,
                           -2.01248513e-01, 4.27121085e-01, 1.13829315e+00,
                           2.23507587e-01, 3.63918039e-01, -4.81166520e-01,
                           -1.83058614e+00, -1.92091976e+00, -6.99839864e-01,
                           -8.68590357e-02, -1.17292516e+00, 5.71445248e-01,
                           -1.42467286e+00, -1.90160691e-01, -5.48832450e-01,
                           2.61124951e-01, -4.17376058e-01],
                          [6.48295373e-01, 7.02448555e-01, -5.25544370e-01,
                           -1.10167566e+00, 4.44984380e-01, 1.28376036e+00,
                           1.73071863e-01, 2.75541990e-01, -1.38379826e+00,
                           -1.59916845e+00, -1.52027017e+00, -6.53498015e-01,
                           -6.80350407e-02, -1.85306750e+00, 7.98553221e-01,
                           1.99624747e-01, 8.24792561e-01, -4.06672961e-01,
                           3.83661970e-01, -4.49981813e-01],
                          [4.90749295e-01, 8.89659243e-01, -6.18047050e-01,
                           -1.09924088e+00, 4.50327419e-01, 1.30796153e+00,
                           -2.50735121e-01, 3.82421446e-01, -1.39985025e-01,
                           -6.02929066e-01, -8.52843488e-01, -1.92039836e-01,
                           -3.83091180e-02, -1.37304681e+00, 5.71716691e-01,
                           -2.07487871e+00, 1.56493681e-01, -6.28930598e-01,
                           2.56988018e-01, -1.78757826e+00],
                          [6.34195312e-01, 1.26039693e-01, -6.64359813e-01,
                           -1.13531994e+00, 4.17950197e-01, 1.21031113e+00,
                           9.93106532e-03, 4.38036504e-01, -3.88117022e-01,
                           -1.47547628e+00, -1.86157805e+00, -6.58908485e-01,
                           -1.71239956e-01, -1.42058032e+00, 4.51919497e-01,
                           -6.08145804e-02, 3.79502620e-01, -6.06991951e-01,
                           2.79557870e-01, -3.12144072e-01],
                          [4.80239286e-01, 6.45955713e-01, -1.09960151e+00,
                           -5.38299998e-01, 4.62528687e-01, 9.16774429e-01,
                           -7.39652526e-02, 3.93848480e-01, -4.38855658e-01,
                           -4.34490888e-01, -9.70335601e-01, -5.04827717e-01,
                           3.90945682e-02, -1.42421778e+00, 6.46906622e-01,
                           -1.45379988e+00, 3.90941642e-01, -5.40145334e-01,
                           2.64162194e-01, -1.71618782e+00],
                          [4.23474830e-01, 9.32568796e-01, -7.37575746e-01,
                           -1.01596022e+00, -1.50644857e-01, 6.83480410e-01,
                           2.32974486e-01, 8.90872022e-01, -3.32046529e-01,
                           -1.05945665e+00, -1.06139571e+00, -3.86032620e-01,
                           1.48113844e-01, -1.31651214e+00, 1.13610050e-01,
                           -1.62718270e+00, 7.52102774e-01, -5.05764971e-01,
                           3.69051787e-01, -1.74879357e+00],
                          [7.08493792e-01, 1.13536247e-01, -4.84206850e-01,
                           -6.30379077e-01, 5.41079336e-01, 1.42727587e+00,
                           2.34198655e-01, 4.08158481e-01, -3.14846991e-01,
                           -1.91505163e+00, -1.79359941e+00, -4.76756004e-01,
                           -2.05835406e-01, -1.33365133e+00, 6.76403514e-01,
                           -1.39546471e+00, -4.52490765e-02, -5.19973558e-01,
                           3.38679363e-01, -3.71081589e-01],
                          [6.83727632e-01, 7.42800586e-01, -9.60478952e-01,
                           -6.36632043e-01, 4.83103375e-01, 1.15182498e+00,
                           -2.63548080e-01, 4.14029250e-01, 3.07317619e-01,
                           -9.07379349e-01, -9.95204020e-01, -4.37706527e-01,
                           -1.80688293e-01, -1.42458769e+00, 9.22874478e-01,
                           -1.39148916e+00, 3.37854180e-01, -5.90648395e-01,
                           2.10382058e-01, -1.40049186e+00],
                          [5.86119824e-01, 6.77782666e-01, -5.24745874e-01,
                           -9.44576799e-01, 4.01203358e-01, 5.97865539e-01,
                           1.96902334e-01, 2.93049464e-01, -4.41378257e-01,
                           -1.18718976e+00, -9.14940080e-01, -2.71785455e-01,
                           -4.35866079e-01, -8.65467782e-01, 7.92038570e-01,
                           -1.32147074e+00, 3.68531558e-01, -4.99728162e-01,
                           2.37193576e-01, -5.49415107e-01],
                          [-8.07496989e-01, 6.67950411e-01, -4.47046079e-01,
                           -9.50885102e-01, -1.57503087e-01, 9.39544341e-01,
                           -4.31585550e-01, 1.24626325e+00, -4.19076189e-01,
                           -9.43057886e-01, -1.09951736e+00, -3.14128262e-01,
                           -4.92110456e-02, -1.66946852e+00, 4.36718645e-01,
                           -1.94506472e+00, 2.44470161e-01, -5.36317114e-01,
                           -1.25410567e-02, -1.52540137e+00],
                          [6.74830545e-01, 2.68010640e-01, -5.87458514e-01,
                           -6.50023351e-01, 4.37647669e-01, 1.31680926e+00,
                           1.83844542e-01, 3.23608916e-01, -3.24937387e-01,
                           -1.42185992e+00, -1.49778435e+00, -6.48479608e-01,
                           -1.23925590e-01, -1.85306750e+00, 7.98553221e-01,
                           -2.43819563e+00, 4.85469563e-01, -3.48366221e-01,
                           3.26268563e-01, -4.49981813e-01],
                          [6.51781367e-01, 1.23766340e-01, -4.83654046e-01,
                           -6.41390938e-01, 4.16594500e-01, 1.21967932e+00,
                           6.25856113e-03, 3.39176939e-01, -6.31031825e-01,
                           -1.43073027e+00, -1.83127432e+00, -3.33731413e-01,
                           -2.72482523e-01, -8.41916805e-01, 4.97522055e-01,
                           -1.41558588e+00, 7.79348445e-01, -5.69887661e-01,
                           2.47666826e-01, -2.98550419e-01],
                          [4.94027169e-01, 8.11569539e-01, -1.27914024e+00,
                           -9.90727011e-01, 4.73055272e-01, 1.38167099e+00,
                           -6.48247978e-02, 5.84176731e-01, -4.48659394e-01,
                           -5.77697836e-01, -1.17285686e+00, -5.75555886e-01,
                           -1.82495711e-02, -1.71015869e+00, 8.13392149e-01,
                           -2.04485922e+00, 3.44457615e-01, -6.36292560e-01,
                           1.52569721e-01, -5.57495251e-01],
                          [6.01780778e-01, 8.04181139e-01, -5.21859004e-01,
                           -6.30821765e-01, 4.80790716e-01, 1.51041858e+00,
                           -2.44272517e-02, 3.01436277e-01, -3.42136924e-01,
                           -1.44985081e+00, -1.63493294e+00, -8.37767636e-01,
                           -2.04963252e-01, -1.18531717e+00, 4.43957146e-01,
                           -6.53580702e-02, 5.34501372e-01, -5.12758835e-01,
                           3.54808167e-01, -4.15569908e-01],
                          [6.32946598e-01, 1.37292795e-01, -5.59326889e-01,
                           -1.09824483e+00, 4.18109691e-01, 1.21187250e+00,
                           1.87517046e-01, 3.59619797e-01, -4.06004541e-01,
                           -1.52515027e+00, -1.52890710e+00, -6.53027539e-01,
                           -2.23923036e-02, 1.13611860e+00, 4.17174692e-01,
                           1.00966111e-01, 5.92164444e-01, -4.56734303e-01,
                           3.32971442e-01, -3.12144072e-01],
                          [7.47932342e-01, 3.35699750e-01, -8.76636881e-01,
                           -9.70086688e-01, 4.98972999e-01, 1.24349015e+00,
                           3.38579385e-01, 4.19637932e-01, 4.26166424e-01,
                           -6.80101152e-01, -1.10733528e+00, -2.92486383e-01,
                           -1.52561320e-01, -8.69475147e-01, 4.85488047e-01,
                           -1.67253646e+00, 5.80673426e-01, -5.79310973e-01,
                           3.13857762e-01, -4.97036766e-01],
                          [-7.76487259e-01, 5.37914572e-01, -1.08215744e+00,
                           -7.80892956e-01, -1.39560045e-01, 9.15017893e-01,
                           -1.80223041e-01, 8.78553890e-01, -7.34057056e-01,
                           -6.25499191e-01, -1.11217494e+00, 6.66433465e-02,
                           -2.82003540e-01, -1.21669792e+00, 4.33732763e-01,
                           -1.90685072e+00, 2.74419601e-01, -6.14280293e-01,
                           6.32558743e-01, -1.32282742e+00],
                          [4.74672103e-01, 4.08276571e-01, -6.40773472e-01,
                           -9.94434522e-01, 4.76643880e-01, 9.85604622e-01,
                           -1.67246859e-01, 3.66014742e-01, 4.49156473e-01,
                           -7.87629561e-01, -1.62205199e+00, -7.68607719e-01,
                           -3.61951010e-01, -8.71694611e-01, 8.56099305e-01,
                           -2.11138890e+00, 2.37866726e-01, -5.79973550e-01,
                           2.65680815e-01, -5.87439312e-01],
                          [4.00529711e-01, 6.79146679e-01, -1.10881492e+00,
                           -7.40054999e-01, -5.51293042e-01, 1.04526179e+00,
                           -2.74075925e-01, 1.17261654e+00, 3.91824680e-01,
                           -1.31245888e+00, -9.41074257e-01, -2.37754385e-01,
                           -2.83384451e-01, -9.53568162e-01, 7.05900407e-01,
                           -2.21694390e+00, 2.27311628e-01, -4.76906080e-01,
                           4.34719061e-01, -1.62702106e+00],
                          [6.83519513e-01, 9.51494467e-01, -8.43345743e-01,
                           -6.46813864e-01, 4.50327419e-01, 1.30809164e+00,
                           -2.47470673e-01, 4.09259250e-01, 3.32428944e-01,
                           -5.84005643e-01, -1.15647646e+00, -1.91569361e-01,
                           -3.03443998e-01, -1.61570818e+00, 9.42508913e-01,
                           -2.06968615e+00, 1.56285698e-01, -6.28930598e-01,
                           3.64234092e-01, -1.85991931e+00],
                          [4.23995128e-01, 6.55731134e-01, -8.65703629e-01,
                           -6.10845474e-01, 4.27360326e-01, 1.33398428e+00,
                           5.07188133e-01, 3.82998040e-01, -3.21497479e-01,
                           -5.82428691e-01, -1.02394417e+00, -2.18856947e-01,
                           2.43105974e-01, -1.33186343e+00, 6.34601170e-01,
                           -2.07536551e+00, 2.35318943e-01, -6.15531827e-01,
                           2.57302215e-01, -1.32824587e+00],
                          [6.48139284e-01, 2.40957729e-01, -5.11785671e-01,
                           -1.10344641e+00, 4.38126151e-01, 1.28662287e+00,
                           1.46416111e+00, 3.48926609e-01, -1.15596172e+00,
                           -1.60557482e+00, -1.51937669e+00, -6.28170744e-01,
                           -9.12924863e-02, -1.85208107e+00, 6.16142992e-01,
                           5.31624039e-01, 7.64685698e-01, -5.33887667e-01,
                           3.29829468e-01, -4.55875565e-01],
                          [6.82478918e-01, 2.11404129e-01, -5.26035752e-01,
                           -1.10532784e+00, 4.34856530e-01, 1.35259056e+00,
                           1.66182345e+00, 1.46856819e-01, -8.68957850e-02,
                           -1.54555208e+00, -1.51610061e+00, -6.13507587e-01,
                           -2.97120880e-01, -2.00090844e+00, 6.04199465e-01,
                           5.47201718e-01, 1.61387712e+00, 1.03952732e-01,
                           3.50074957e-02, -4.97702189e-01],
                          [6.15620691e-01, 8.19014773e-01, -1.36040256e+00,
                           -1.91011356e-01, 5.72659090e-01, 1.20400061e+00,
                           1.01009169e-01, 3.57418258e-01, -8.87591596e-01,
                           -1.48533223e+00, -1.56062551e+00, -6.00961570e-01,
                           -1.04665517e-01, -1.94079797e+00, 4.15274585e-01,
                           -1.80000054e-01, 1.09683331e+00, -6.05372319e-01,
                           2.83046161e-02, -6.68525927e-01],
                          [-8.35905232e-01, 7.92075530e-01, -7.99059931e-01,
                           -4.70070728e-01, -1.59098024e-01, 3.20007549e-01,
                           -2.14076371e-02, 1.15468973e+00, -4.27847954e-01,
                           -7.08486286e-01, -9.89247513e-01, 9.45582337e-02,
                           -9.36182308e-02, -1.40769510e+00, 4.13374479e-01,
                           -2.07301263e+00, 3.67907611e-01, -5.92488885e-01,
                           5.48091986e-01, -1.36607996e+00],
                          [-9.15718866e-01, 6.67893577e-01, -5.79964937e-01,
                           -8.41098507e-01, -2.75687921e-01, 9.78643534e-01,
                           -4.21139315e-01, 1.25302512e+00, -3.31129220e-01,
                           -7.47121609e-01, -1.08648750e+00, -3.86738334e-01,
                           -1.21018409e-01, -1.34838610e+00, 3.52661550e-01,
                           -1.96145373e+00, 1.66268845e-01, -6.02501154e-01,
                           7.67244731e-01, -1.38765869e+00],
                          [1.64523738e+00, 3.39507617e-01, -5.87704205e-01,
                           -8.94719077e-01, 5.05432494e-01, 1.23672423e+00,
                           2.88207418e+00, 4.09835843e-01, -1.54908582e+00,
                           -1.42964612e+00, -1.83015748e+00, -8.02638789e-01,
                           -6.44165576e-01, -1.96638345e+00, 8.01319821e-02,
                           1.17201270e+00, 7.59070178e-01, 4.64173539e-01,
                           1.16227545e-01, -2.50164619e-01],
                          [4.91009444e-01, 2.32887323e-01, -1.11753696e+00,
                           -5.25904737e-01, 5.09021102e-01, 1.28363025e+00,
                           1.53648396e-01, 4.33109251e-01, -3.79689248e-01,
                           -4.79434018e-01, -1.08127556e+00, -5.17216908e-01,
                           -3.89496547e-01, -1.52224409e+00, 6.46273253e-01,
                           -2.08834691e+00, 1.68868623e-01, -6.34083971e-01,
                           2.63900362e-01, -9.49619858e-01],
                          [4.73475419e-01, 4.05946383e-01, -6.25049245e-01,
                           -5.76149811e-01, 4.68509701e-01, 1.17101677e+00,
                           1.69236136e-01, 3.85409249e-01, 2.09453162e-02,
                           -6.73891903e-01, -1.20301168e+00, -2.82606395e-01,
                           -1.46601599e-01, -5.81684681e-01, 9.60695647e-01,
                           -1.46069625e+00, 1.28520071e-01, -5.02083990e-01,
                           3.19199119e-01, -6.53601427e-01],
                          [7.01625866e-01, 7.06313257e-01, -5.93355100e-01,
                           -5.33907615e-02, 5.30154018e-01, 1.28590724e+00,
                           1.24758030e-01, 3.75764413e-01, -4.44015519e-01,
                           -1.92411911e+00, -1.88242583e+00, -6.92155429e-01,
                           -4.71760191e-02, -1.85454714e+00, 8.05791722e-01,
                           5.30974969e-01, 4.92021003e-01, -4.04611612e-01,
                           2.75996965e-01, -4.61198953e-01],
                          [-8.49068758e-01, 5.83836320e-01, -1.20144045e+00,
                           -3.31564758e-01, -1.57263846e-01, 1.26073022e+00,
                           -4.96451138e-02, 5.50629477e-01, -5.98754026e-01,
                           -6.99615932e-01, -1.12282220e+00, -3.27772055e-01,
                           -2.63542943e-01, -1.39339189e+00, 5.23218734e-01,
                           -2.02076250e+00, 3.81322464e-01, -5.92930603e-01,
                           4.28435111e-01, -6.87062727e-01]])

Obj = np.array([[969.95047595, 38.44690923],
       [544.89020575, 68.47024662],
       [882.18056124, 52.35236809],
       [927.91483356, 44.22955261],
       [978.97906439, 35.02463916],
       [207.1466561, 71.44857509],
       [893.40874325, 49.65978564],
       [926.9443722, 44.26220691],
       [567.06041473, 67.5095817],
       [737.10632396, 62.92172049],
       [685.79694885, 63.96968891],
       [578.83275374, 64.75306252],
       [406.95368655, 70.66800152],
       [839.35719256, 55.42984519],
       [477.61284708, 69.871305],
       [752.55606206, 61.84663886],
       [896.60107081, 49.54618887],
       [822.06211508, 57.95387432],
       [876.35132915, 54.67045821],
       [943.12797675, 41.61206047],
       [758.0787866, 61.30145861],
       [983.1467759, 29.6932501],
       [746.57334628, 62.32331649],
       [980.248058, 34.77466076],
       [968.29808351, 40.20869282],
       [486.19556983, 69.62109418],
       [912.86378457, 47.04885182],
       [774.6873648, 60.50473401],
       [763.67012955, 60.68200982],
       [889.00521257, 50.6698169],
       [932.60703743, 43.67944869],
       [224.80226539, 70.9609576],
       [982.49246273, 31.51583148],
       [892.06886675, 50.46834936],
       [985.89933726, 28.24913242],
       [917.04211673, 46.33089789],
       [980.71455106, 33.85529942],
       [800.36825027, 59.95226913],
       [976.71298126, 37.41275511],
       [780.0047134, 60.0073251],
       [696.10133657, 63.06281797],
       [923.86187676, 44.53746122],
       [823.69344881, 57.63824527],
       [886.76842584, 51.36384336],
       [517.89093902, 69.0536543],
       [912.02928469, 49.33139673],
       [923.46106082, 46.32907272],
       [859.86367432, 55.40594074],
       [972.4470894, 37.66687719],
       [985.19087764, 28.29175329],
       [885.19237605, 52.30047747],
       [315.19163965, 70.83861553],
       [832.90021063, 56.36999708],
       [592.33896927, 64.57164568],
       [809.34076279, 58.11758451],
       [804.27241464, 58.88225226],
       [986.05615164, 28.00480638],
       [988.56293486, 27.0184251],
       [950.37030065, 41.24710406],
       [268.92487676, 70.942057],
       [439.93912968, 70.25123068],
       [990.22022148, 18.15832283],
       [830.41356724, 57.03341062],
       [868.53742435, 54.74928295],
       [982.77662607, 30.59026085],
       [570.81185429, 66.61144675]])


def compute_distance(virtual_samples, pareto_solvers, num_samples):
    """

    :param virtual_samples: dataframe，索引为化学式
    :param pareto_solvers:
    :param num_samples:每个帕累托非支配解对应的最接近的虚拟样本数量
    :return:
    """
    m = pareto_solvers.shape[0]
    n = len(virtual_samples.iloc[:, 0])
    distance_matrix = np.ones((n, m))
    distance_matrix_rank = np.ones((n, m))
    filtered_samples_distance = np.ones((num_samples, m))
    filtered_samples_index = [list(np.ones(num_samples)) for i in range(m)]

    for i in range(n):
        for j in range(m):
            a = np.array(virtual_samples.iloc[i, :])
            b = np.array(pareto_solvers[j, :])
            distance_matrix[i, j] = np.linalg.norm(a - b)
    distance_matrix_df = pd.DataFrame(distance_matrix, index=virtual_samples.index)

    for i in range(m):
        distance_matrix_rank[:, i] = distance_matrix_df.iloc[:, i].rank(method="first")
        for j in range(n):
            if distance_matrix_rank[j, i] in range(1, num_samples + 1):
                rank_ = int(distance_matrix_rank[j, i])
                filtered_samples_distance[rank_ - 1, i] = distance_matrix[j, i]
                filtered_samples_index[i][rank_ - 1] = virtual_samples.index[j]

    return filtered_samples_index, filtered_samples_distance
