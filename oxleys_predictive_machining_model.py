from numpy import float_power,log,sin,radians,cos,sqrt,arange,log10,arctan,pi
import matplotlib.pyplot as plt

#All temperatures are in degree celsius (except T_mod)
#All angles are in radians

def plot_graph(x = [],y = [],save = False,title = "plot",xlabel = "x-axis",ylabel = "y-axis"):
	if len(x)==0 or len(y)==0:
		return
	elif len(x) != len(y):
		return
	else:
		plt.plot(x,y)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if save:
			plt.savefig(title + ".png")
		plt.show()
		

class Material:
	def __init__(self,name = "0.2% Carbon Steel"):
		self.name = name #name of the material
		self.rho = None #density
		self.n = None #strain-hardening index
		self.sigma_l = None #value of uni-axial (effective) flow stress at epsilon = 1
		self.S = None #specific heat
		self.K = None #thermal conductivity
		self.CarbonPercentage = None #percentage of carbon in the steel material

		if name=="0.2% Carbon Steel":
			self.CarbonPercentage = 0.2

	def calculate_n(self,T_mod = 20):
		if self.name == "0.2% Carbon Steel":
			n_16 = None
			n = None

			#calculating n for 0.16% Carbon Steel
			if T_mod<=73:
				n_16 = 0.04768
			elif T_mod>73 and T_mod<=396:
				n_16 = 0.04937 + (3.5861 * float_power(10,-4) * T_mod) \
					   - (1.4026 * float_power(10,-5) * float_power(T_mod,2)) \
					   + (1.7680 * float_power(10,-7) * float_power(T_mod,3)) \
					   - (9.4992 * float_power(10,-10) * float_power(T_mod,4)) \
					   + (2.7341 * float_power(10,-12) * float_power(T_mod,5)) \
					   - (4.1361 * float_power(10,-15) * float_power(T_mod,6)) \
					   + (2.5569 * float_power(10,-18) * float_power(T_mod,7))
			elif T_mod>396 and T_mod <=528:
				n_16 = 0.19109
			elif T_mod>528 and T_mod<=693:
				n_16 = -145.26 + (0.81927 * T_mod) \
					   - (0.88538 * float_power(10,-3) * float_power(T_mod,2)) \
					   - (2.5350 * float_power(10,-6) * float_power(T_mod,3)) \
					   + (5.0364 * float_power(10,-9) * float_power(T_mod,4)) \
					   + (2.4501 * float_power(10,-12) * float_power(T_mod,5)) \
					   - (1.04279 * float_power(10,-14) * float_power(T_mod,6)) \
					   + (5.8410 * float_power(10,-18) * float_power(T_mod,7)) 
			elif T_mod>693 and T_mod<=827:
				n_16 = -21.227 + (0.08507 * T_mod) \
					   - (4.4837 * float_power(10,-5) * float_power(T_mod,2)) \
					   - (1.3310 * float_power(10,-7) * float_power(T_mod,3)) \
					   - (3.5910 * float_power(10,-11) * float_power(T_mod,4)) \
					   + (5.1253 * float_power(10,-13) * float_power(T_mod,5)) \
					   - (5.1724 * float_power(10,-16) * float_power(T_mod,6)) \
					   + (1.5471 * float_power(10,-19) * float_power(T_mod,7))
			elif T_mod>827 and T_mod<=974:
				n_16 = -65.632 + (0.30193 * T_mod) \
					   - (0.49548 * float_power(10,-3) * float_power(T_mod,2)) \
					   + (2.7300 * float_power(10,-7) * float_power(T_mod,3)) \
					   + (9.1267 * float_power(10,-11) * float_power(T_mod,4)) \
					   - (1.0362 * float_power(10,-13) * float_power(T_mod,5)) \
					   - (3.1959 * float_power(10,-17) * float_power(T_mod,6)) \
					   + (3.0674 * float_power(10,-20) * float_power(T_mod,7))
			else:
				n_16 = 0.18388

			#calculating the scaling factor
			NFAC = (0.244 - (0.3396 * self.CarbonPercentage)) / 0.189
			
			#calculating the n for C percentage Carbon Steel based on 0.16% Carbon Steel
			n = NFAC * n_16 

			return n

	def calculate_sigma_l(self,T_mod = 20):
		if self.name == "0.2% Carbon Steel":
			sigma_l_16 = None
			sigma_l = None

			#calculating the sigma_l for 0.16% carbon steel
			if T_mod <= 458:
				sigma_l_16 = 1126.62 - (0.98421 * T_mod)
			elif T_mod>458 and T_mod <=748:
				sigma_l_16 = -19914.15 + (135.07 * T_mod) \
							 - (0.20137 * float_power(T_mod,2)) \
							 - (3.1090 * float_power(10,-4) * float_power(T_mod,3)) \
							 + (7.2551 * float_power(10,-7) * float_power(T_mod,4)) \
							 + (7.3255 * float_power(10,-10) * float_power(T_mod,5)) \
							 - (2.2977 * float_power(10,-12) * float_power(T_mod,6)) \
							 + (1.2673 * float_power(10,-15) * float_power(T_mod,7))
			elif T_mod>748 and T_mod<=1200:
				sigma_l_16 = 17756.97 - (97.198 * T_mod) \
							 + (0.23022 * float_power(T_mod,2)) \
							 - (2.4637 * float_power(10,-4) * float_power(T_mod,3)) \
							 + (2.8921 * float_power(10,-8) * float_power(T_mod,4)) \
							 + (1.8495 * float_power(10,-10) * float_power(T_mod,5)) \
							 - (1.6072 * float_power(10,-13) * float_power(T_mod,6)) \
							 + (4.2722 * float_power(10,-17) * float_power(T_mod,7))
			else:
				sigma_l_16 = 172.42

			#rescaling function
			SIGFAC = (531.31 + (753.17 * self.CarbonPercentage)) / 651.72

			#calculating the sigma_l for C percentage Carbon Steel based on 0.16% Carbon Steel
			if T_mod>=200 and T_mod<=700:
				sigma_l = sigma_l_16 * (1 + (SIGFAC - 1) * ((1400 - T_mod) / 900))
			elif T_mod>700 and T_mod<=1100:
				sigma_l = sigma_l_16 * (1 + self.CarbonPercentage * ((T_mod-700) / T_mod))
			else:
				sigma_l = sigma_l_16 * SIGFAC

			#sigma_l is in MPa, hence converting it into Pa
			sigma_l = sigma_l * float_power(10,6)

			return sigma_l

	def calculate_S(self,T = 20):
		if self.name=="0.2% Carbon Steel":
			self.S = 420.0 + (0.504 * T)

 	def calculate_K(self,T = 20):
 		if self.name=="0.2% Carbon Steel":
			self.K = 54.17 - (0.0298 * T)

	def make_T_mod_graphs(self):
		Y1 = []
		Y2 = []

		X = range(200,1100,1)

		for T_mod in X:
			sigma_l = self.calculate_sigma_l(T_mod)
			n = self.calculate_n(T_mod)
			Y1.append(sigma_l/float_power(10,6))
			Y2.append(n)

		plot_graph(X,Y1,save = True,title = "Effective Flow Stress at $\epsilon=1$ versus Velocity-Modified Temperature" \
			,xlabel = "Velocity-Modified Temperature, $T_{mod}$ (Kelvin)",ylabel = "Effective Flow Stress at $\epsilon=1$, $\sigma_l$ (MPa)")
		
		plot_graph(X,Y2,save = True,title = "Strain Hardening Index versus Velocity-Modified Temperature" \
			,xlabel = "Velocity-Modified Temperature, $T_{mod}$ (Kelvin)",ylabel = "Strain Hardening Index, $n$")		


class MachiningProcess:
	M_PER_MIN_TO_M_PER_SEC = 1.0 / 60
	MM_TO_M = 1.0 / 1000

	def __init__(self,material = None,U = 250,t1 = 0.5,w = 4,T_W = 20,alpha = 5):
		#constant properties
		self.U = U #cutting velocity (m/min)
		self.t1 = t1 #undeformed chip thickness (mm)
		self.w = w #width of cut (mm)
		self.T_W = T_W #initial work temperature (degree celsius)
		self.alpha = alpha #rake angle (radians)

		#constants used in calculating T_mod
		self.nu = 0.09
		self.epsilon_0 = 1.0

		self.neta = 1.0 #chip formation zone temperature factor
		self.psi = 1.0 #tool-chip interface temperature factor

		#varying properties
		self.material = material
		self.t2 = 0 #chip thickness
		self.V = 0 #chip velocity
		self.Vs = 0 #shear velocity (velocity discontinuity)
		self.phi = 0 #shear angle (radians)
		self.lam = 0 #mean friction angle
		self.theta = 0 #angle made by R with AB (radians)
		self.C = 0 #strain-rate constant for chip formation zone
		self.R_T = 0 #thermal number
		self.beta = 0 #proportion of heat conducted into work
		self.neta = 0 #chip formation zone temperature factor
		self.N = 0 #normal force at tool-chip interface
		self.F = 0 #frictional force in the tool-chip interface
		self.F_C = 0 #force in the cutting direction
		self.F_N = 0 #force normal to AB
		self.F_T = 0 #force normal to cutting direction and machined surfce
		self.F_S = 0 #shear force on AB
		self.R = 0 #resultant force
		self.T_C = 0 #mean chip temperature
		self.delta_T_C = 0 #rise in mean chip temperature
		self.delta_T_M = 0 #maximum temperature rise in chip
		self.sigma_N = 0 #normal stress acting on the tool-chip interface at B 
						 #determined through parallel sided shear zone theory
		self.sigma_N_dash = 0 #normal stress acting on the tool-chip interface at B
							  #based on the uniform normal stress assumed along the too-chip interface

		#properties in chip formation zone (AB)
		self.l = 0 #length (mm)
		self.gamma_dot_AB = 0 #shear strain-rate
		self.gamma_AB = 0 #shear
		self.epsilon_AB = 0 #effective strain
		self.epsilon_dot_AB = 0 #effective strain-rate
		self.T_AB = 0 #average temperature
		self.delta_T_SZ = 0 #temperature rise
		self.T_mod = 0 #velocity-modified temperature (always in Kelvin)
		self.k_AB = 0 #shear flow stress


		#properties in tool-chip interface zone
		self.delta = 0 #ratio of tool-chip interface plastic zone thickness to chip thickness
		self.h = 0 #length (mm)
		self.tau_int = 0 #resolved shear stress
		self.gamma_dot_int = 0 #shear strain-rate
		self.epsilon_dot_int = 0 #effective strain-rate
		self.T_int = 0 #temperature
		self.k_chip = 0 #shear flow stress in the chip

	def calculate_l(self):
		return (self.t1 / sin(self.phi))

	def calculate_Vs(self):
		return (self.U * cos(self.alpha) / cos(self.phi - self.alpha))

	def calculate_gamma_dot_AB(self):
		return (self.C * self.Vs * MachiningProcess.M_PER_MIN_TO_M_PER_SEC / (self.l * MachiningProcess.MM_TO_M))

	def calculate_gamma_AB(self):
		return (0.5 * cos(self.alpha)) / (sin(self.phi) * cos(self.phi - self.alpha))

	def calculate_epsilon_dot_AB(self):
		return (self.gamma_dot_AB * sqrt(3))

	def calculate_epsilon_AB(self):
		return (self.gamma_AB / sqrt(3))

	def calculate_T_mod(self,T,epsilon_dot):
		return (T * (1.0 - self.nu * log10(epsilon_dot / self.epsilon_0)))

	def calculate_k_AB(self):
		return (self.material.sigma_l * float_power(self.epsilon_AB,self.material.n) / sqrt(3))

	def calculate_R_T(self):
		return (self.material.rho * self.material.S * self.U * MachiningProcess.M_PER_MIN_TO_M_PER_SEC
			   * self.t1 * MachiningProcess.MM_TO_M / self.material.K)

	def calculate_beta(self):
		c = self.R_T * tan(self.phi)
		beta = None

		#what happens when c <0.04?
		if c>=0.04 and c<=10:
			beta = 0.5 - 0.35 * log10(c)
		elif c>10:
			beta = 0.3 - 0.15 * log10(c)

		return beta

	def calculate_F_S(self):
		return (self.k_AB * self.w * MM_TO_M * self.l * MM_TO_M)

	def calculate_delta_T_SZ(self):
		return ((1.0 - self.beta) * self.F_S * cos(self.alpha)
			   / (self.material.rho * self.material.S * self.t1 * MM_TO_M
			   * self.w * MM_TO_M * cos(self.phi - self.alpha)))

	def calculate_T_AB(self):
		return (self.T_W + (self.neta * self.delta_T_SZ))

	def calculate_theta(self,max_iter = 1000,t0 = 0,epsilon = 1e-8):
		"""
		t0 is in radians
		tan(theta) = 1 + 2 (pi/4 - theta) - Cn
		A closed form solution cannot be obtained
		Hence, an iterative method has to be used
		Solution is found using Newton's Method
		f(theta) = 1 + 2 (pi/4 - theta) - Cn - tan(theta)
		f'(theta) = -2 - (sec(theta))^2
		"""
		#constant for holding the constant part of the equation to avoid multiple calculations
		a = 1 + (pi/2) - (self.C * self.material.n)
		f = lambda t: a - (2 * t)
		Df = lambda t: -2 - power(sec(t),2)

		t = t0
		for i in range(0,max_iter):
			f_theta = f(t)
			if abs(f_theta)<epsilon:
				return t
			Df_theta = Df(t)
			if Df_theta == 0:
				print("No solution found for theta")
				return None
			t = t - (f_theta / Df_theta)

		print("Maximum iterations exceeded for theta")
		return None

	def calculate_lam(self):
		return (self.theta - self.phi + self.alpha)

	def calculate_R(self):
		return (self.F_S / cos(self.theta))

	def calculate_F(self):
		return (self.R * sin(self.lam))

	def calculate_N(self):
		return (self.R * cos(self.lam))

	def calculate_F_C(self):
		return (self.R * cos(self.lam - self.alpha))

	def calculate_t2(self):
		return (self.t1 * cos(self.phi - self.alpha) / sin(self.phi))

	def calculate_h(self):
		return (((self.t1 * sin(self.theta)) / (cos(self.lam) * sin(self.phi)))
			   * (1.0 + ((self.C * self.material.n)
			   /  (3.0 * (1.0 + 2.0 * (pi/4 - self.phi) - (self.C * self.material.n))))))

	def calculate_tau_int(self):
		return (self.F / (self.h * MachiningProcess.MM_TO_M * self.w * MachiningProcess.MM_TO_M))

	def calculate_V(self):
		return (self.U * sin(self.phi) / cos(self.phi - self.alpha))

	def calculate_gamma_dot_int(self):
		return (self.V / (self.delta * self.t2 * MachiningProcess.MM_TO_M))

	def calculate_epsilon_dot_int(self):
		return (self.gamma_dot_int / sqrt(3))

	def calculate_delta_T_C(self):
		return (self.F * sin(self.phi) / (self.material.rho * self.material.S * self.t1 * MachiningProcess.MM_TO_M
			   * self.w * MachiningProcess.MM_TO_M * cos(self.phi - self.alpha)))

	def calculate_T_C(self):
		return (self.T_W + self.delta_T_SZ + self.delta_T_C)

	def calculate_delta_T_M(self):
		return (self.delta_T_C * power(10, 0.06 - (0.195 * self.delta * sqrt(self.R_T * self.t2 / self.h))
			   + (0.5 * log10(self.R_T * self.t2 / self.h))))

	def calculate_T_int(self):
		return (self.T_W + self.delta_T_SZ + (self.psi * self.delta_T_M))

	def calculate_k_chip(self):
		return (self.material.sigma_l / sqrt(3))

	def start_machining(self):
		Delta = arange(0.01,1,0.01)
		Phi = arange(5.0,45.1,0.1)

		self.C = 5.0
		self.phi = radians(5.0)
		

		for delta in Delta:
			self.delta = delta
			
			for phi in Phi:
				self.phi = phi

				self.l = self.calculate_l()
				self.Vs = self.calculate_Vs()
				self.gamma_dot_AB = self.calculate_gamma_dot_AB()
				self.gamma_AB = self.calculate_gamma_AB()
				self.epsilon_dot_AB = self.calculate_epsilon_dot_AB()
				self.epsilon_AB = self.calculate_epsilon_AB()

				self.T_AB = self.T_W
				last_T_AB = 0
				while abs(self.T_AB - last_T_AB)>0.1:
					last_T_AB = self.T_AB
					self.material.S = self.material.calculate_S(self.T_AB)
					self.material.K = self.material.calculate_K(self.T_AB)
					self.T_mod = self.calculate_T_mod(self.T_AB,self.epsilon_dot_AB)
					self.material.n = self.material.calculate_n(self.T_mod)
					self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
					self.k_AB = self.calculate_k_AB()
					self.R_T = self.calculate_R_T()
					self.beta = self.calculate_beta()
					self.F_S = self.calculate_F_S()
					self.delta_T_SZ = self.calculate_delta_T_SZ()
					self.T_AB = self.calculate_T_AB()

				self.theta = self.calculate_theta()
				self.lam = self.calculate_lam()
				self.R = self.calculate_R()
				self.F = self.calculate_F()
				self.N = self.calculate_N()
				self.F_C = self.calculate_F_C()

				self.t2 = self.calculate_t2()
				self.h = self.calculate_h()
				self.tau_int = self.calculate_tau_int()
				self.V = self.calculate_V()
				self.gamma_dot_int = self.calculate_gamma_dot_int()
				self.epsilon_dot_int = self.calculate_epsilon_dot_int()

				self.T_C = self.T_W + self.delt_T_SZ
				last_T_C = 0
				while abs(self.T_C - last_T_C)>0.1:
					last_T_C = self.T_C
					self.material.S = self.material.calculate_S(self.T_C)
					self.material.K = self.material.calculate_K(self.T_C)
					self.delta_T_C = self.calculate_delta_T_C()
					self.T_C = self.calculate_T_C()

				self.R_T = self.calculate_R_T()
				self.delta_T_M = self.calculate_delta_T_M()
				self.T_int = self.calculate_T_int()
				self.T_mod = self.calculate_T_mod(self.T_int,self.epsilon_dot_int)
				self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
				self.k_chip = self.calculate_k_chip() 


			


if __name__ == '__main__':
	#defining the material properties of the material used for machining
	#the material is taken as 0.2% Carbon Steel
	CarbonSteel = Material("0.2% Carbon Steel")
	#taking the density to be 7862 kg/m3
	CarbonSteel.rho = 7862
	#plotting the variation of n and sigma_l with T_mod
	# CarbonSteel.make_T_mod_graphs()
	
	Process1 = MachiningProcess(material = CarbonSteel,U = 250,t1 = 0.5,w = 4,T_W = 20,alpha = 5)
	Process1.start_machining()


