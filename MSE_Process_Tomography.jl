using Distributed, SharedArrays, PyPlot, Statistics

addprocs(20)

@everywhere using Distributions, LinearAlgebra, Optim, BenchmarkTools, HDF5


#------------------->CSPSA-MLE<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
@everywhere function cspsa(Psi,Guess,It,Nex)
	a=3.; b=0.07; A=0.; r=1/6; s=1. #gain coeficients
	Dim=size(Psi)[1]

	Data=zeros(It) #mse vector
	FunPl=zeros(2*Dim,Dim,It)
	FunMi=zeros(2*Dim,Dim,It)
	GuessPl=zeros(Complex{Float64},Dim,Dim,It)
	GuessMi=zeros(Complex{Float64},Dim,Dim,It)
	GuessIn=zeros(Complex{Float64},Dim,Dim,It+1)
	Para=zeros(Int(2*Dim))
	GuessIn[:,:,1]=Guess

   	for i=1:It
		for j=1:Dim
			ck=b/(1*i+1)^r; ak=a/(1*i+1+A)^s
			Step = ck*rand([1 -1 1im -im],Dim) #Step for guess
			@views GuessPl[:,j,i] .= normalize(GuessIn[:,j,i]+Step )#Plus perturbation
			@views GuessMi[:,j,i] .= normalize(GuessIn[:,j,i]-Step )#Minus perturbation
			@views FunPl[:,j,i] .= 4*Dim*SE(Psi[:,j],GuessPl[:,j,i],Nex,Dim) #Plus Measure
			@views FunMi[:,j,i] .= 4*Dim*SE(Psi[:,j],GuessMi[:,j,i],Nex,Dim) #Minus Measure
			#Gradient estimation
			@views Grad = sum(FunPl[1:Dim,j,i]-FunMi[1:Dim,j,i])./(2*conj(Step))
			#next guess generation
			@views GuessIn[:,j,i+1] .= GuessIn[:,j,i]-ak*Grad
			#Parametrizations
			@views Para[1:Dim] .= vec(real(GuessIn[:,j,i+1]))
			@views Para[Dim+1:2*Dim] .= vec(imag(GuessIn[:,j,i+1]))
			#Target function for mle
			Target(x)=KLD(FunPl[:,j,1:i]/4,FunMi[:,j,1:i]/4,GuessPl[:,j,1:i],GuessMi[:,j,1:i],x,Dim,i)
			#Optimization of MLE
			Res=Optim.minimizer(optimize(Target,Para,NelderMead()))
			#New Guess
			GuessIn[:,j,i+1]=normalize(  Res[1:Dim]+1im*Res[Dim+1:2*Dim] )
		end
		GuessIn[:,:,i+1]=GramSchmidt(GuessIn[:,:,i+1],Dim)
		@views Data[i]=sum(abs2.(GuessIn[:,:,i+1]-Psi)) #saving data
	end

	return Data
end
#------------------->MAIN PROGRAM<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

function main()
	It=3*10^1 #Iterations
	Repe=10^2 #Repetitions for estimation
	M=10^2 #Guess
	N=10^2 #Target
	Dim=4 #Dimension of the transformation
	Ini=3
	Fin=5
	Tot=Fin-Ini+1
	Nex=Dim*[10^i for i=Ini:Fin]

	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	NameArch="grsctom"*string(Dim)*"n"*string(Ini)*"_"*string(Fin)*"It"*string(It)*".h5"
	Data=h5open(NameArch,"w")
	Data["Info"]=[It;Repe;M;N;Ini;Fin;Dim]
	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Guess=randn(M,Dim,Dim)+1im*randn(M,Dim,Dim)
	for i=1:M
        Guess[i,:,:]=GramSchmidt(Guess[i,:,:],Dim)
    end
	#state generator
	Psi=randn(N,Dim,Dim)+1im*randn(N,Dim,Dim)
	for i=1:N
        Psi[i,:,:]=GramSchmidt(Psi[i,:,:],Dim)
    end

	GMB=zeros(Tot,It)
	for i=1:Tot
		GMB[i,:]=[1/j for j=1:It]*(2*Dim-1)/(2*Nex[i])
	end
    #Data matrix
	infi=SharedArray{Float64}(It,Repe,M,N,Tot)

	Data["Bound"]=GMB

	println("Dimension=========="*string(Dim))
	#cspsa ejecution
    @time @sync @distributed for l=1:N
		for h=1:M
            for n=1:Tot
				for i=1:Repe
					infi[:,i,h,l,n]= cspsa(Psi[l,:,:],Guess[h,:,:],It,Nex[n])
				end
			end
        end
	end
	for n=1:Tot
        @views Data["N"*string(n)]=infi[:,:,:,:,n]
	end
	close(Data)
    Graphic(NameArch)
end
#------------------->GRAM-SCHMIDT<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
@everywhere function GramSchmidt(Mat,Dim)
    Aux=Mat
    for i=2:Dim
        for j=1:i-1
            @views Mat[:,i] .= Aux[:,i]-dot(Mat[:,j],Aux[:,i])*Mat[:,j]/(norm(Mat[:,j])^2)
        end
    end
    return Mat./sqrt.(sum( abs2.(Mat) ,dims=1))
end


#------------------->UNITARY PROJECTION<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
@everywhere function UProj(Mat)
    return Mat*(Mat'*Mat)^(-.5)
end


#------------------->PROBABILITY SIMULATION<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

@everywhere function SE(Psi,Guess,Nex,Dim)
	#probability calculation
	prob=[  vec(abs2.(Psi-Guess))   ;   vec(abs2.(Psi+Guess))  ]./4
	return rand(Multinomial(Nex,prob))/Nex #Experiment simulation
end
#------------------->LIKELIHOOD<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
@everywhere function KLD(DataPl,DataMi,GuessPl,GuessMi,Para,Dim,It) #Kullback-Leibler Divergence
	@views Mat = normalize( Para[1:Dim] + 1im*Para[Dim+1:2*Dim] )

    ProbPl=zeros(2*Dim,It)
    ProbMi=zeros(2*Dim,It)
	#Theoretical (model) log probabilities
    @views ProbPl[1:Dim,:] = log.( abs2.(Mat.-GuessPl) )[:,:]
    @views ProbPl[Dim+1:2*Dim,:] = log.( abs2.(Mat.+GuessPl) )[:,:]
    @views ProbMi[1:Dim,:] =log.( abs2.(Mat.-GuessMi) )[:,:]
    @views ProbMi[Dim+1:2*Dim,:] =log.( abs2.(Mat.+GuessMi) )[:,:]
	#KL divergence (model dependent part)
    return  -sum(DataPl.*ProbPl+DataMi.*ProbMi)
end
#------------------->GRAPHIC FUNCTION<------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
function Graphic(NameArch)
    #-------------------------------------------------------------------------------------
        #READING
    #-------------------------------------------------------------------------------------
    File=h5open(NameArch,"r")
    (It,Repe,M,N,Ini,Fin,Dim)=File["Info"][:]
    Tot=Fin-Ini+1
    GMB=File["Bound"][:,:]

    MSEData=zeros(It,Repe,M,N,Tot)

    for i=1:Tot
        @views MSEData[:,:,:,:,i].=File["N"*string(i)][:,:,:,:]
    end
    close(File)
#-------------------------------------------------------------------------------------
    #DATA ANALIZER
#-------------------------------------------------------------------------------------



    RepeMean=mean(MSEData,dims=2)[:,1,:,:,:]
    GuessMean=mean(RepeMean,dims=2)[:,1,:,:]
    MSEMean=mean(GuessMean,dims=2)[:,1,:]

    GuessMed=median(RepeMean,dims=2)[:,1,:,:]
    MSEMed=median(GuessMean,dims=2)[:,1,:]

    GuessVar=var(RepeMean,dims=2)[:,1,:,:]
    MSEVar=var(GuessMean,dims=2)[:,1,:]

    GuessVP=GuessMean+GuessVar/2
    GuessVM=GuessMean-GuessVar/2

    MSEVP=MSEMean+MSEVar/2
    MSEVM=MSEMean-MSEVar/2


    iqrpg=zeros(It,N,Tot);iqrmg=zeros(It,N,Tot)
    for l=1:Tot
        for i=1:M
            for j=1:It
                iqrpg[j,i,l],iqrmg[j,i,l]=quantile(RepeMean[j,:,i,l],[.75 .25])
            end
        end
    end


    iqrp=zeros(It,Tot) ;iqrm=zeros(It,Tot)
    for j=1:Tot
        for l=1:It
            iqrp[l,j],iqrm[l,j]=quantile(GuessMean[l,:,j],[.75 .25])
        end
    end
#-------------------------------------------------------------------------------------
    #MEAN AND MEDIAN GRAPHIC
#-------------------------------------------------------------------------------------
    rc("xtick", labelsize=15)
    rc("ytick", labelsize=15)
    x=1:It
    Colors=["deepskyblue" "tomato" "gold" "mediumpurple" "greenyellow"]
    Points=[ "1" "o" "^"]

    Cols=2
    Rows=1

    fig,ax=subplots(nrows=Rows,ncols=Cols,figsize=(20,7))

    for k=1:Tot
        ax[1].plot(x,MSEMean[:,k],color=Colors[k],lw=.3,marker=Points[k],label="N=10^"*string(2+k))
        ax[1].fill_between(x,MSEVP[:,k],MSEVM[:,k],color=Colors[k],alpha=.3,lw=0)
        ax[2].plot(x,MSEMed[:,k],color=Colors[k],lw=.3,marker=Points[k],label="N=10^"*string(2+k))
        ax[2].fill_between(x,iqrp[:,k],iqrm[:,k],color=Colors[k],alpha=.3,lw=0)

        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].grid()
        ax[1].legend()
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        ax[2].grid()
        ax[2].legend()
    end

    fig.text(0.5, 0.06 , "k",ha="center", size=25)
    fig.text(.06,.5,"MSE",ha="center",rotation="vertical", size=25)
    fig.savefig("meanmediangrscDim="*string(Dim)*".pdf")
    clf()
#-------------------------------------------------------------------------------------
    #4 RANDOM RECONSTRUCTIONS GRAPHIC
#-------------------------------------------------------------------------------------
    Cols=2
    Rows=4

    Rnum=rand(1:M,4)


    Colors=["deepskyblue" "tomato" "gold" "mediumpurple" "greenyellow" "dimgray"]
    Points=[ "1" "o" "^"]
    fig,ax=subplots(nrows=Rows,ncols=Cols,figsize=(20,35))

    for i=1:4
        for k=1:Tot
            ax[i].plot(x,GuessMean[:,Rnum[i],k,1],color=Colors[k],lw=.3,marker=Points[k],label="N=10^"*string(2+k))
            ax[i].fill_between(x,GuessVP[:,Rnum[i],k,1],GuessVM[:,Rnum[i],k,1],color=Colors[k],alpha=.3,lw=0)
            ax[i+4].plot(x,GuessMed[:,Rnum[i],k,1],color=Colors[k],lw=.3,marker=Points[k],label="N=10^"*string(2+k))
            ax[i+4].fill_between(x,iqrpg[:,Rnum[i],k,1],iqrmg[:,Rnum[i],k,1],color=Colors[k],alpha=.3,lw=0)
        end
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].grid()
        ax[i].legend()
        ax[i+4].set_xscale("log")
        ax[i+4].set_yscale("log")
        ax[i+4].grid()
        ax[i+4].legend()
    end

    fig.text(0.5, 0.06 , "k",ha="center", size=25)
    fig.text(.06,.5,"MSE",ha="center",rotation="vertical", size=25)
    fig.savefig("4randgrscDim="*string(Dim)*".pdf")
    clf()
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
end


main()
