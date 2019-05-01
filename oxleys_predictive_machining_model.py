from numpy import float_power,log,sin,radians,cos,sqrt,arange,log10,arctan,pi,tan,degrees
import matplotlib.pyplot as plt

#All temperatures are in degree celsius (except T_mod)
#All angles are in radians

def plot_graph(x = [],y1 = [],y2 = [],save = False,title = "plot",xlabel = "x-axis",ylabel = "y-axis"):
	if len(y1)==0 or len(x)==0:
		return
	elif len(y1) != len(x):
		return
	else:
		if len(y2)!=0:
			plt.plot(x,y1,'r')
			plt.plot(x,y2,'b')
		else:
			plt.plot(x,y1)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if save:
			plt.savefig(title + ".png")
		# plt.show()
		plt.close()

# def save_graph(title = "plot",):
# 	plt.savefig(title + ".png")

# def show_graph():
# 	plt.show()
		

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
			return (420.0 + (0.504 * T))

 	def calculate_K(self,T = 20):
 		if self.name=="0.2% Carbon Steel":
 			K = 54.17 - (0.0298 * T)
 			if K>0:
				return K
			else:
				print "The Temperature is greater than the threshold. Equation for K is no longer valid. Taking K as 1.0"
				return 1.0

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
	M_PER_MIN_TO_M_PER_SEC = lambda self,x: x / 60
	MM_TO_M = lambda self,x: x * 1e-3
	CELSIUS_TO_KELVIN = lambda self,T: T + 273.15

	def __init__(self,material = None,U = 250,t1 = 0.5,w = 4,T_W = 20,alpha = 5):
		#constant properties
		self.U = U #cutting velocity (m/min)
		self.t1 = t1 #undeformed chip thickness (mm)
		self.w = w #width of cut (mm)
		self.T_W = T_W #initial work temperature (degree celsius)
		self.alpha = radians(alpha) #rake angle (radians)

		#constants used in calculating T_mod
		self.nu = 0.09
		self.epsilon_0 = 1.0

		self.neta = 1.0 #chip formation zone temperature factor
		self.psi = 1.0 #tool-chip interface temperature factor

		#varying properties
		self.material = material
		self.t2 = None #chip thickness
		self.V = None #chip velocity
		self.Vs = None #shear velocity (velocity discontinuity)
		self.phi = None #shear angle (radians)
		self.lam = None #mean friction angle
		self.theta = None #angle made by R with AB (radians)
		self.C = None #strain-rate constant for chip formation zone
		self.R_T = None #thermal number
		self.beta = None #proportion of heat conducted into work
		self.N = None #normal force at tool-chip interface
		self.F = None #frictional force in the tool-chip interface
		self.F_C = None #force in the cutting direction
		self.F_N = None #force normal to AB
		self.F_T = None #force normal to cutting direction and machined surfce
		self.F_S = None #shear force on AB
		self.R = None #resultant force
		self.T_C = None #mean chip temperature
		self.delta_T_C = None #rise in mean chip temperature
		self.delta_T_M = None #maximum temperature rise in chip
		self.sigma_N = None #normal stress acting on the tool-chip interface at B 
						 #determined through parallel sided shear zone theory
		self.sigma_N_dash = None #normal stress acting on the tool-chip interface at B
							  #based on the uniform normal stress assumed along the too-chip interface

		#properties in chip formation zone (AB)
		self.l = None #length (mm)
		self.gamma_dot_AB = None #shear strain-rate
		self.gamma_AB = None #shear
		self.epsilon_AB = None #effective strain
		self.epsilon_dot_AB = None #effective strain-rate
		self.T_AB = None #average temperature
		self.delta_T_SZ = None #temperature rise
		self.T_mod = None #velocity-modified temperature (always in Kelvin)
		self.k_AB = None #shear flow stress


		#properties in tool-chip interface zone
		self.delta = None #ratio of tool-chip interface plastic zone thickness to chip thickness
		self.h = None #length (mm)
		self.tau_int = None #resolved shear stress
		self.gamma_dot_int = None #shear strain-rate
		self.epsilon_dot_int = None #effective strain-rate
		self.T_int = None #temperature
		self.k_chip = None #shear flow stress in the chip

	def print_status(self,i=0,j=0,k=0,m=0):
		print "***************************************************"
		print "delta min loop,i = ",i,"; delta loop,j = ",j,"; C loop,k = ",k,"; phi loop,m = ",m

		print "S=",self.material.S
		print "K=",self.material.K
		print "sigma_l=",self.material.sigma_l
		print "n=",self.material.n

		print "t2=",self.t2
		print "V=",self.V
		print "Vs=",self.Vs
		print "phi=",degrees(self.phi)
		print "lam=",degrees(self.lam)
		print "theta=",degrees(self.theta)
		print "C=",self.C
		print "R_T=",self.R_T
		print "beta=",self.beta
		print "N=",self.N
		print "F=",self.F
		print "F_C=",self.F_C
		print "F_N=",self.F_N
		print "F_T=",self.F_T
		print "F_S=",self.F_S
		print "R=",self.R
		print "T_C=",self.T_C
		print "delta_T_C=",self.delta_T_C
		print "delta_T_M=",self.delta_T_M
		print "sigma_N=",self.sigma_N				
		print "sigma_N_dash=",self.sigma_N_dash

		print "l=",self.l
		print "gamma_dot_AB=",self.gamma_dot_AB
		print "gamma_AB=",self.gamma_AB
		print "epsilon_AB=",self.epsilon_AB
		print "epsilon_dot_AB=",self.epsilon_dot_AB
		print "T_AB=",self.T_AB
		print "delta_T_SZ=",self.delta_T_SZ
		print "T_mod=",self.T_mod
		print "k_AB=",self.k_AB

		print "delta=",self.delta
		print "h=",self.h
		print "tau_int=",self.tau_int
		print "gamma_dot_int=",self.gamma_dot_int
		print "epsilon_dot_int=",self.epsilon_dot_int
		print "T_int=",self.T_int
		print "k_chip=",self.k_chip
		print "***************************************************"

	def calculate_l(self):
		return (self.t1 / sin(self.phi))

	def calculate_Vs(self):
		return (self.U * cos(self.alpha) / cos(self.phi - self.alpha))

	def calculate_gamma_dot_AB(self):
		return (self.C * self.M_PER_MIN_TO_M_PER_SEC(self.Vs) / self.MM_TO_M(self.l))

	def calculate_gamma_AB(self):
		return ((0.5 * cos(self.alpha)) / (sin(self.phi) * cos(self.phi - self.alpha)))

	def calculate_epsilon_dot_AB(self):
		return (self.gamma_dot_AB * sqrt(3))

	def calculate_epsilon_AB(self):
		return (self.gamma_AB / sqrt(3))

	def calculate_T_mod(self,T,epsilon_dot):
		if epsilon_dot<=0:
			print "Negative value ecountered in T_mod's log function"
			return None
		return (self.CELSIUS_TO_KELVIN(T) * (1.0 - (self.nu * log10(epsilon_dot / self.epsilon_0))))

	def calculate_k_AB(self):
		return (self.material.sigma_l * float_power(self.epsilon_AB,self.material.n) / sqrt(3))

	def calculate_F_S(self):
		return (self.k_AB * self.MM_TO_M(self.w) * self.MM_TO_M(self.l))

	def calculate_R_T(self):
		return (self.material.rho * self.material.S * self.M_PER_MIN_TO_M_PER_SEC(self.U)
			   * self.MM_TO_M(self.t1) / self.material.K)

	def calculate_beta(self):
		c = self.R_T * tan(self.phi)
		beta = None

		#what happens when c <0.04?
		#assuming same for c>=0.04 and c<=10 to avoid errors
		if c<=0:
			print "R_T*tan(phi) = ",c," : Non-Positive value, unable to calculate beta"
			return
		elif c<0.04:
			beta = 0.5 - 0.35 * log10(c)
		elif c>=0.04 and c<=10:
			beta = 0.5 - 0.35 * log10(c)
		elif c>10:
			beta = 0.3 - 0.15 * log10(c)

		if beta<0:
			beta = 0

		return beta

	def calculate_delta_T_SZ(self):
		return ((1.0 - self.beta) * self.F_S * cos(self.alpha)
			   / (self.material.rho * self.material.S * self.MM_TO_M(self.t1)
			   * self.MM_TO_M(self.w) * cos(self.phi - self.alpha)))
 
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
		f = lambda t: a - (2 * t) - tan(t)
		Df = lambda t: -2 - float_power(1.0/cos(t),2)

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

	def calculate_V(self):
		return (self.U * sin(self.phi) / cos(self.phi - self.alpha))

	def calculate_h(self):
		return (((self.t1 * sin(self.theta)) / (cos(self.lam) * sin(self.phi)))
			   * (1.0 + ((self.C * self.material.n)
			   /  (3.0 * (1.0 + 2.0 * (pi/4 - self.phi) - (self.C * self.material.n))))))

	def calculate_tau_int(self):
		return (self.F / (self.MM_TO_M(self.h) * self.MM_TO_M(self.w)))

	def calculate_gamma_dot_int(self):
		return (self.M_PER_MIN_TO_M_PER_SEC(self.V) / (self.delta * self.MM_TO_M(self.t2)))

	def calculate_epsilon_dot_int(self):
		return (self.gamma_dot_int * sqrt(3))

	def calculate_delta_T_C(self):
		return (self.F * sin(self.phi) / (self.material.rho * self.material.S * self.MM_TO_M(self.t1)
			   * self.MM_TO_M(self.w) * cos(self.phi - self.alpha)))

	def calculate_T_C(self):
		return (self.T_W + self.delta_T_SZ + self.delta_T_C)

	def calculate_delta_T_M(self):
		return (self.delta_T_C * float_power(10, 0.06 - (0.195 * self.delta
			   * sqrt(self.R_T * self.t2 / self.h))
			   + (0.5 * log10(self.R_T * self.t2 / self.h))))

	def calculate_T_int(self):
		return (self.T_W + self.delta_T_SZ + (self.psi * self.delta_T_M))

	def calculate_k_chip(self):
		return (self.material.sigma_l / sqrt(3))

	def find_min_index(self,list1,list2,end = "last"):
		"""
		function to find the index at which the difference between list1 and list2 is minimum (zero)
		tries to find the right-most point since it is required in tau_int and k_chip for finding phi
		"""
		minimum = None
		index = None

		diff_list = [x-y for x,y in zip(list1,list2)]

		if end == "last":
			for i in range(1,len(diff_list)):
				if (diff_list[i]>0) != (diff_list[i-1]>0):
					index = i
		elif end == "first":
			for i in range(1,len(diff_list)):
				if (diff_list[i]>0) != (diff_list[i-1]>0):
					index = i
					break

		if index:
			return index
		else:
			print "The two lists never crosses each other"
			return

	def calculate_sigma_N_dash(self):
		return (self.k_AB * (1.0 + (pi/2) - (2 * self.alpha) - (2 * self.C * self.material.n)))

	def calculate_sigma_N(self):
		return (self.N / (self.MM_TO_M(self.h) * self.MM_TO_M(self.w)))

	def calculate_all_properties(self,delta = 0.05,C = 5.9,phi = 0.01):
		"""
		The default values of C and delta are based on Stevenson and Duncan's (1983) work
		phi is i randians
		"""
		MAX_ITER = 1000
		num = 0

		self.delta = delta
		self.C = C
		self.phi = phi

		self.l = self.calculate_l()
		self.Vs = self.calculate_Vs()
		self.gamma_dot_AB = self.calculate_gamma_dot_AB()
		self.gamma_AB = self.calculate_gamma_AB()
		self.epsilon_dot_AB = self.calculate_epsilon_dot_AB()
		self.epsilon_AB = self.calculate_epsilon_AB()

		self.T_AB = self.T_W
		last_T_AB = 0

		for num in range(0,MAX_ITER):				
			if abs(self.T_AB - last_T_AB)<0.1:
				break							
			last_T_AB = self.T_AB
			self.material.S = self.material.calculate_S(self.T_AB)
			self.material.K = self.material.calculate_K(self.T_AB)
			self.T_mod = self.calculate_T_mod(self.T_AB,self.epsilon_dot_AB)
			self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
			self.material.n = self.material.calculate_n(self.T_mod)				
			self.k_AB = self.calculate_k_AB()
			self.F_S = self.calculate_F_S()							
			self.R_T = self.calculate_R_T()
			self.beta = self.calculate_beta()							
			self.delta_T_SZ = self.calculate_delta_T_SZ()
			self.T_AB = self.calculate_T_AB()													

		if num + 1 == MAX_ITER:
			print "MAX_ITER exceeded in T_AB loop" 

		self.theta = self.calculate_theta()
		self.lam = self.calculate_lam()
		self.R = self.calculate_R()
		self.F = self.calculate_F()
		self.N = self.calculate_N()
		self.F_C = self.calculate_F_C()

		self.t2 = self.calculate_t2()
		self.V = self.calculate_V()
		self.h = self.calculate_h()
		self.tau_int = self.calculate_tau_int()			
		self.gamma_dot_int = self.calculate_gamma_dot_int()
		self.epsilon_dot_int = self.calculate_epsilon_dot_int()

		self.T_C = self.T_W + self.delta_T_SZ
		last_T_C = 0

		for num in range(0,MAX_ITER):								
			if abs(self.T_C - last_T_C)<0.1:					
				break
			last_T_C = self.T_C
			self.material.S = self.material.calculate_S(self.T_C)
			self.material.K = self.material.calculate_K(self.T_C)				
			self.delta_T_C = self.calculate_delta_T_C()				
			self.T_C = self.calculate_T_C()

		if num + 1 == MAX_ITER:
			print "MAX_ITER exceeded in T_C loop"

		self.material.S = self.material.calculate_S(self.T_C)
		self.material.K = self.material.calculate_K(self.T_C)
		self.R_T = self.calculate_R_T()						
		self.delta_T_M = self.calculate_delta_T_M()
		self.T_int = self.calculate_T_int()
		self.T_mod = self.calculate_T_mod(self.T_int,self.epsilon_dot_int)
		self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
		self.k_chip = self.calculate_k_chip() 

		self.sigma_N_dash = self.calculate_sigma_N_dash()
		self.sigma_N = self.calculate_sigma_N()

	def start_machining(self):
		"""
		The while loop runs two times:
		i = 0: For finding the delta_min at which F_C is minimum
		i = 1: For finding the value of all properties at delta_min
		"""
		
		#predefined lists for iterations
		Delta = arange(0.01,0.3,0.01)
		Phi = arange(5.0,45.1,0.1)
		C_list = arange(1.0,7,0.2)
		
		#lists for storing data for plotting and use after iterations
		T_mod_list = []
		k_chip_list = []
		F_C_list = []		
		
		min_delta = 0
		MAX_ITER = 1000
		num = 0
		min_delta_flag = True
		i = 0

		while min_delta_flag == True:		
			for j,delta in enumerate(Delta):
				self.delta = delta
				#lists for storing data for plotting and use after iterations
				Sigma_N = []
				Sigma_N_dash = []
				Phi_eq_list = []
				Phi_eq_list = [] #list of phi at which tau_int = k_chip for differenct C
				for k,C in enumerate(C_list):
					self.C = C
					Tau_int = []
					K_chip = []
					for m,phi in enumerate(Phi):
						self.phi = radians(phi)

						self.l = self.calculate_l()
						self.Vs = self.calculate_Vs()
						self.gamma_dot_AB = self.calculate_gamma_dot_AB()
						self.gamma_AB = self.calculate_gamma_AB()
						self.epsilon_dot_AB = self.calculate_epsilon_dot_AB()
						self.epsilon_AB = self.calculate_epsilon_AB()

						self.T_AB = self.T_W
						last_T_AB = 0

						for num in range(0,MAX_ITER):				
							if abs(self.T_AB - last_T_AB)<0.1:
								break							
							last_T_AB = self.T_AB
							self.material.S = self.material.calculate_S(self.T_AB)
							self.material.K = self.material.calculate_K(self.T_AB)
							self.T_mod = self.calculate_T_mod(self.T_AB,self.epsilon_dot_AB)
							self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
							self.material.n = self.material.calculate_n(self.T_mod)				
							self.k_AB = self.calculate_k_AB()
							self.F_S = self.calculate_F_S()							
							self.R_T = self.calculate_R_T()
							self.beta = self.calculate_beta()							
							self.delta_T_SZ = self.calculate_delta_T_SZ()
							self.T_AB = self.calculate_T_AB()													

						if num + 1 == MAX_ITER:
							print "MAX_ITER exceeded in T_AB loop" 

						self.theta = self.calculate_theta()
						self.lam = self.calculate_lam()
						self.R = self.calculate_R()
						self.F = self.calculate_F()
						self.N = self.calculate_N()
						self.F_C = self.calculate_F_C()

						self.t2 = self.calculate_t2()
						self.V = self.calculate_V()
						self.h = self.calculate_h()
						self.tau_int = self.calculate_tau_int()			
						self.gamma_dot_int = self.calculate_gamma_dot_int()
						self.epsilon_dot_int = self.calculate_epsilon_dot_int()

						self.T_C = self.T_W + self.delta_T_SZ
						last_T_C = 0

						for num in range(0,MAX_ITER):								
							if abs(self.T_C - last_T_C)<0.1:					
								break
							last_T_C = self.T_C
							self.material.S = self.material.calculate_S(self.T_C)
							self.material.K = self.material.calculate_K(self.T_C)				
							self.delta_T_C = self.calculate_delta_T_C()				
							self.T_C = self.calculate_T_C()

						if num + 1 == MAX_ITER:
							print "MAX_ITER exceeded in T_C loop"

						self.material.S = self.material.calculate_S(self.T_C)
						self.material.K = self.material.calculate_K(self.T_C)
						self.R_T = self.calculate_R_T()						
						self.delta_T_M = self.calculate_delta_T_M()
						self.T_int = self.calculate_T_int()
						self.T_mod = self.calculate_T_mod(self.T_int,self.epsilon_dot_int)
						self.material.sigma_l = self.material.calculate_sigma_l(self.T_mod)
						self.k_chip = self.calculate_k_chip() 

						Tau_int.append(self.tau_int)
						K_chip.append(self.k_chip)

					self.phi = radians(Phi[self.find_min_index(Tau_int,K_chip,end = "last")])
					Phi_eq_list.append(self.phi)					

					# calculating all the properties for the value of phi at which tau_int = k_chip for the	generated C				
					self.calculate_all_properties(delta = self.delta,C = self.C,phi = self.phi)
					
					self.sigma_N_dash = self.calculate_sigma_N_dash()
					self.sigma_N = self.calculate_sigma_N()
					Sigma_N.append(self.sigma_N)
					Sigma_N_dash.append(self.sigma_N_dash)

					# plot_graph(x = Phi
					# 	      ,y1 = [y/(1e+6) for y in Tau_int]
					# 		  ,y2 = [y/(1e+6) for y in K_chip]
					# 		  ,save = True
					# 		  ,title = "Resolved Shear Stress at Tool-Chip Interface, Shear Flow Stress in the Chip VS. Shear Angle"
					# 		  + " j = " + str(j) + " k = " + str(k) + " m = " + str(m)
					# 		  ,xlabel = "Shear Angle, $\phi$ (degrees)"
					# 		  ,ylabel = "Stress (MPa)")

				index = self.find_min_index(Sigma_N,Sigma_N_dash,end = "first")
				self.C = C_list[index]
				self.phi = Phi_eq_list[index]
				
				# calculating all the properties for the value of phi at which tau_int = k_chip and C at which sigma_N = sigma_N_dash					
				self.calculate_all_properties(delta = self.delta,C = self.C,phi = self.phi)

				F_C_list.append(self.F_C)
				T_mod_list.append(self.T_mod)#storing the T_mod values at tool-chip interface

				plot_graph(x = C_list
					      ,y1 = [y/(1e+6) for y in Sigma_N_dash]
						  ,y2 = [y/(1e+6) for y in Sigma_N]
						  ,save = True
						  ,title = "$\sigma_N^{'}$ , $\sigma_N$ VS C"
						  + "; j = " + str(j) + "; k = " + str(k) + "; m = " + str(m) +" )"
						  ,xlabel = "C"
						  ,ylabel = "Stress (MPa)")

			if i==0:
				plot_graph(x = Delta
					      ,y1 = F_C_list
						  ,save = True
						  ,title = "$F_C$ VS $\delta$"
						  ,xlabel = "Delta, $\delta$"
						  ,ylabel = "Cutting Force, $F_C$ (N)")
				plot_graph(x = Delta
					      ,y1 = T_mod_list
						  ,save = True
						  ,title = "$T_{mod}$ VS $\delta$"
						  ,xlabel = "Delta, $\delta$"
						  ,ylabel = "Velocity-Modified Temperature, $T_{mod}$ (K)")
				# plot_graph(x = Delta
				# 	      ,y1 = T_mod_list
				# 		  ,save = True
				# 		  ,title = "$T_{mod}$ VS $\delta$"
				# 		  ,xlabel = "Delta, $\delta$"
				# 		  ,ylabel = "Velocity-Modified Temperature, $T_{mod}$ (K)")

			min_F_C = min(F_C_list)
			min_delta = Delta[F_C_list.index(min_F_C)]
			Delta = [min_delta]
			i += 1
			if i==2:
				min_delta_flag = False

		self.print_status()

		
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


