import numpy as np

X = 0 
Y = 1
Z = 2

# print p 
kz = [0,0,0,1]
kx = [0,1,0,0]
k0 = [1,0,0,0]

##############################

def dot4(x,y):
	if (np.size(x)!=4 or np.size(y)!=4):
		print("ERROR!")
		return 0
	return x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3]

def dot3(x,y):
	if (np.size(x)!=4 or np.size(y)!=4):
		print("ERROR!")
	return x[1]*y[1] + x[2]*y[2] + x[3]*y[3]

def get_3vec(x):
	return x[1:]

def get_direction(x):
	if (np.size(x)!=4):
		print("ERROR! Wrong size np.shape(x):", np.shape(x))
		return 0
	return get_3vec(x)/np.sqrt(dot3(x,x))


def dot4_df(df1, df2):
	return df1['t']*df2['t'] - df1['x']*df2['x'] - df1['y']*df2['y'] - df1['z']*df2['z']

def dot3_df(df1, df2):
	return df1['x']*df2['x'] + df1['y']*df2['y'] + df1['z']*df2['z']

def costheta(df1, df2):
	return dot3_df(df1, df2)/np.sqrt(dot3_df(df1, df1) * dot3_df(df2, df2))

def inv_mass(df):
	return np.sqrt(dot4_df(df, df))




def R(v4, theta, i):
	c, s = np.cos(theta), np.sin(theta)
	if i ==X:
		R = np.array( ((1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, c, -s), (0, 0, s, c)) )
	if i ==Y:
		R = np.array( ((1.0, 0, 0, 0), (0, c, 0, -s), (0, 0, 1.0, 0), (0, s, 0, c)) )
	if i ==Z:
		R = np.array( ((1.0, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1.0)) )
	return R.dot(v4)

def L(v4,beta):
	gamma = 1.0/np.sqrt(1.0 - beta*beta)
	R = np.array( ((gamma, 0, 0, -gamma*beta), (0, 1, 0, 0), (0, 0, 1, 0), (-gamma*beta, 0, 0, gamma)) )
	return R.dot(v4)

def T(v4, beta, theta, phi):
	return L( R( R(v4,-phi, Z), theta, Y), -beta)

def Tinv(v4, beta, theta, phi):
	return R( R( L(v4, beta), -theta, Y), phi, Z)


def cos_opening_angle(theta1, phi1, theta2, phi2):
    aux = np.sin(theta1)*np.cos(phi1) * np.sin(theta2)*np.cos(phi2) +\
           np.sin(theta1)*np.sin(phi1) * np.sin(theta2)*np.sin(phi2) +\
           np.cos(theta1)*np.cos(theta2)
    #remove values >1 or <-1 becuase of numerical precision
    return np.where(aux>1,1, np.where(aux<-1,-1, aux))

def inv_mass_massless(energy1, energy2, cos_open_angle):
    return np.sqrt(2 * energy1 * energy2 * (1 - cos_open_angle))