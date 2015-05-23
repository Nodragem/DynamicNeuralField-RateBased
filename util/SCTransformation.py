import numpy as np
import matplotlib.pyplot as plt

def r(x,y):
    return (np.sqrt(x**2+y**2))

def phi(x,y):
    return ( 2*np.arctan2(y,(x + np.sqrt(x**2+y**2))) )

def x(r, phi):
    return (r*np.cos(phi))

def y(r, phi):
    return (r*np.sin(phi))

def uColi(r, phi):
    return( 1.4* np.log(np.sqrt(r**2+2*3.0*r*np.cos(phi) + 3.0**2)/3.0) )

def vColi(r,phi):
    return( 1.8*np.arctan2((r*np.sin(phi)),(r*np.cos(phi)+3.0))  )

def rColi(u,v):
    return( 3.0*np.sqrt( np.exp(2*u/1.4) - 2*np.exp(u/1.4)*np.cos(v/1.8) + 1 ) )

def phiColi(u,v):
    return( np.arctan2( (np.exp(u/1.4)*np.sin(v/1.8)), ( np.exp(u/1.4)*np.cos(v/1.8)-1) ) )
    #return( np.arctan( (np.exp(u/1.4)*np.sin(v/1.8)) / ( np.exp(u/1.4)*np.cos(v/1.8)-1) ) )

def toColi(x,y):
    re, phie = r(x,y), phi(x,y)
    if(type(x) != np.ndarray):
        return(  ( uColi(re,phie), vColi(re,phie) )  )
    else:
        if len(x.shape)>1 or len(y.shape)>1:

            result = np.zeros((np.shape(x)[0], np.shape(x)[1],2))
            result[:,:,0] = uColi(re,phie)
            result[:,:,1] = vColi(re,phie)
        else:
            result = np.zeros((np.shape(x)[0], 1,2))
            #print result.shape, result[:,:,0].shape, np.matrix(uColi(re,phie)).shape
            result[:,:,0] = uColi(re,phie).reshape(np.shape(x)[0],1)
            result[:,:,1] = vColi(re,phie).reshape(np.shape(x)[0],1)
        return(result)


def fromColi(u,v):
    r, phi = rColi(u,v), phiColi(u,v)
    #print np.shape(r), np.shape(phi), np.max(r), np.max(phi)
    #print type(u)
    if (type(u) != float and type(u) != int):
        result = np.zeros((np.shape(u)[0], np.shape(u)[1],2))
        result[:,:,0] = x(r,phi)
        result[:,:,1] = y(r,phi)
        return(result)
    else:
        return (x(r, phi), y(r, phi))

def transfColToCart(m, linx, liny, linu, linv): 
    ##picture tranformation
    start = time.time()
    lx, ly = len(linx), len(liny)
    #print "lu, lv:", lx, ly
    minu, maxu = linu[0], linu[-1]
    minv, maxv = linv[0], linv[-1]
    du = 1/abs(linu[1]-linu[0])
    #print minx, miny, maxx, maxy
    X, Y = np.meshgrid(linx, liny)
    coord = np.zeros((ly, lx, 2))
    coord[:,:,:] = toColi(X,Y)
    coord[:,:,0] = (np.ceil(du*coord[:,:,0])-1)/du
    coord[:,:,1] = (np.trunc(du*coord[:,:,1]))/du
    coord[np.isnan(coord)] = 0
    mt = np.zeros((ly, lx))

    for x in xrange(0, lx):
        for y in xrange(0, ly):
            u, v = coord[y,x,0], coord[y, x, 1]

            if ((u > (maxu)) | (u < (minu)) | (v > (maxv)) | (v < (minv)) ):
                mt[y,x] = 0.0
            else:
                #print u,v
                ind_u = np.nonzero(linu==u)[0][0]

                ind_v = np.nonzero(linv==v)[0][0]
                mt[y, x] = m[ind_v,ind_u]#np.sum(m[ind_y-1:ind_y+1,ind_x-1:ind_x+1])/9

    conn_time = time.time()-start
    a = time.gmtime(conn_time)
    print time.strftime("\ntask time for map transformation: %H:%M:%S",a )
    return(mt)

def transfCartToCol2(m, linx, liny, linu, linv):
    start = time.time()
    lu, lv = len(linu), len(linv)
    print "lu, lv:", lu, lv
    minx, maxx = linx[0], linx[-1]
    miny, maxy = liny[0], liny[-1]
    dx = 1/abs(linx[1]-linx[0])
    #print minx, miny, maxx, maxy
    U, V = np.meshgrid(linu, linv)
    coord = np.zeros((lv, lu, 2))
    coord[:,:,:] = fromColi(U,V)
    coord[:,:,0] = (np.ceil(dx*coord[:,:,0])-1)/dx
    ## transforme: -0.8 en -1, et 0.8 en 0: utile par car on les x<0 n'existe pas
    #(si on transforme -0.8 en 0, on n'echoue la selection)
    coord[:,:,1] = (np.trunc(dx*coord[:,:,1]))/dx
    ## transforme: -0.8 en 0, et 0.8 en 0: utile pour ne pas depasser le y max
    mt = np.zeros((lv, lu))


    for u in xrange(0, lu):
        for v in xrange(0, lv):
            x, y = coord[v,u,0], coord[v, u, 1]

            if ((x > (maxx)) | (x < (minx)) | (y > (maxy)) | (y < (miny)) ):
                mt[v,u] = 0.0
            else:
                ind_x = np.nonzero(linx==x)[0][0]
                ind_y = np.nonzero(liny==y)[0][0]
                mt[v, u] = m[ind_y,ind_x]#np.sum(m[ind_y-1:ind_y+1,ind_x-1:ind_x+1])/9


    conn_time = time.time()-start
    a = time.gmtime(conn_time)
    print time.strftime("\ntask time for map transformation: %H:%M:%S",a )
    return(mt)

def constructSCLine(r, phi): # if r is vector -> ligne, if phi is a vector -> cercle
    x1, y1 = x(r,phi), y(r,phi)
    #print toColi(x1,y1)[:,:,0]
    return(toColi(x1,y1)[:,:,0], toColi(x1,y1)[:,:,1])
    
def xy(r, phi):
    return x(r,phi), y(r,phi)
    
def constructRetinalGrid(r_list, phi_list):
    lines_r = []
    for r in r_list:
        lines_r.append(xy(r, np.radians(np.arange(-90,90,0.1))))
    lines_phi = []
    for phi in phi_list:
        lines_phi.append(xy(np.arange(0,90,0.1), np.radians(phi)))
    return(lines_r, lines_phi)
    
def constructSCLineFromCartesian(x, y):
    return(toColi(x,y)[:,:,0], toColi(x,y)[:,:,1])

def createCircle(xc, yc, r):
    x = np.arange(xc-r,xc+r,0.01)
    root = np.sqrt(r**2 - (x-xc)**2)
    y1 = root + yc
    y2 = - root + yc
    return np.tile(x,2), np.concatenate((y1,y2), axis=0)

def createRectangle(left, down, right, up):
    edge1 = np.arange(down, up, 0.1)
    y1 = np.tile(edge1,2) 
    x1 = np.concatenate( (np.repeat(left, len(edge1)), np.repeat(right, len(edge1)) ) ) 
    edge2 = np.arange(left, right, 0.1)
    x2 = np.tile(edge2,2)
    y2 =  np.concatenate( (np.repeat(up, len(edge2)), np.repeat(down, len(edge2)) ), axis=0 ) 
    #print x1, x2, y1, y2
    #print np.concatenate((x1,x2), axis =0), np.concatenate((y1,y2), axis =0)
    #return x2, y2
    return np.concatenate((x1,x2), axis =0), np.concatenate((y1,y2), axis =0)
    
def constructSCGrid(r_list, phi_list):
    lines_r = []
    for r in r_list:
        lines_r.append(constructSCLine(r, np.radians(np.arange(-90,90,0.1))))

    lines_phi = []
    for phi in phi_list:
        lines_phi.append(constructSCLine(np.arange(0,90,0.1), np.radians(phi)))

    return(lines_r, lines_phi)

def plotGrid(grid, ax):
    lines_r = grid[0]
    lines_phi = grid[1]
    for i in np.arange(0, np.shape(lines_r)[0]):
        ax.plot(lines_r[i][0], lines_r[i][1], color="gray")
    for i in np.arange(0, np.shape(lines_phi)[0]):
        ax.plot(lines_phi[i][0], lines_phi[i][1], color="black")

