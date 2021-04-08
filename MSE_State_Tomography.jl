using Distributed, SharedArrays,PyPlot

addprocs(4)
@everywhere using Distributions, LinearAlgebra, Optim, BenchmarkTools, HDF5


@everywhere function cspsa(Psi,Guess,It,Nex)
	a=3.; b=0.07; A=0.; r=1/6; s=1. #gain coeficients
	Dim=size(Psi)[1]

	Data=zeros(It) #mse vector
	FunPl=zeros(2*Dim,It)
	FunMi=zeros(2*Dim,It)
	GuessPl=zeros(Complex{Float64},Dim,It)
	GuessMi=zeros(Complex{Float64},Dim,It)
	GuessIn=zeros(Complex{Float64},Dim,It+1)
	Para=zeros(2*Dim)
	GuessIn[:,1]=Guess
	Grad=zeros(Complex{Float64},Dim)
    	Step=zeros(Complex{Float64},Dim)
   	for i=1:It
		ck=b/(1*i+1)^r; ak=a/(1*i+1+A)^s
		Step.=ck*rand([1 -1 1im -im],Dim) #Step for guess

        @views GuessPl[:,i].=normalize(GuessIn[:,i]+Step) #Plus perturbation
		@views GuessMi[:,i].=normalize(GuessIn[:,i]-Step) #Minus perturbation

        @views FunPl[:,i].=4*SE(Psi,GuessPl[:,i],Nex) #Plus Measure
		@views FunMi[:,i].=4*SE(Psi,GuessMi[:,i],Nex) #Minus Measure
		#gradient estimation
	    @views Grad.=sum(FunPl[1:Dim,i]-FunMi[1:Dim,i])./(2*conj(Step))
		#next guess generation
		@views GuessIn[:,i+1].=normalize(GuessIn[:,i]-ak*Grad)
		#Parametrizations
 		@views Para[1:Dim].=real(GuessIn[:,i+1])
 		@views Para[Dim+1:2*Dim].=imag(GuessIn[:,i+1])
		#Target function for mle
 		Target(x)=KLD(.25*FunPl,.25*FunMi,GuessPl,GuessMi,x,Dim,i)
		#Optimization of MLE
		Res=Optim.minimizer(optimize(Target,Para,NelderMead()))
		#new guess
     	GuessIn[:,i+1]=normalize(Res[1:Dim]+1im*Res[Dim+1:2*Dim])
		@views Data[i]=norm(GuessIn[:,i+1]-Psi)^2 #saving data
    	end
	return Data
end

function main()
	#guess generator
	It=3*10^1 #Iterations
	Repe=10^1 #Repetitions for estimation
	M=10^1#Guess
	N=10^2#Target
	Dim=2#Dimension of the state
	Ini=3
	Fin=5
	Tot=Fin-Ini+1
	Nex=[10^i for i=Ini:Fin]#Number of photons per measurement

	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	NameArch="cspsamsed"*string(Dim)*"n"*string(Ini)*"_"*string(Fin)*"It"*string(It)*".h5"

	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	#ATENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	#Guess generator
	Guess=randn(M,Dim)+1im*randn(M,Dim)
	Guess=Guess./sqrt.(sum(abs2.(Guess),dims=2))

	#state generator
	Psi=randn(N,Dim)+1im*randn(N,Dim)
	Psi=Psi./sqrt.(sum(abs2.(Psi),dims=2))

	Data=h5open(NameArch,"w")
	GMB=zeros(Tot,It)
	for i=1:Tot
		GMB[i,:]=[1/j for j=1:It]*3/(2*Nex[i])
	end

	#Data matrix
	infi=SharedArray{Float64}(It,Repe,M,N,Tot)

	Data["Bound"]=GMB
	Data["Info"]=[It;Repe;M;N;Ini;Fin;Dim]
#	infi=zeros(It,Repe,M,N,Tot)

	println("Dimension=========="*string(Dim))
	#cspsa ejecution
	for n=1:Tot
		println(string(n))
		@time @sync @distributed for l=1:N
			for h=1:M
				for i=1:Repe
					@views infi[:,i,h,l,n].=cspsa(Psi[l,:],Guess[h,:],It,Nex[n])
				end
			end
			#Data["N"*string(n)*"s"*string(l)]=infi[:,:,:,:l,n]
		end
        Data["N"*string(n)]=infi[:,:,:,:,n]
	end
	close(Data)
	Graphic(NameArch)
end



@everywhere function SE(Psi,Guess,Nex)
	#probability calculation
	prob=.25*[abs2.(Psi-Guess) ;abs2.(Psi+Guess)]
	return rand(Multinomial(Nex,prob))/Nex #Experiment simulation
end

@everywhere function KLD(DataPl,DataMi,GuessPl,GuessMi,Para,Dim,It) #Kullback-Leibler Divergence
	State=zeros(Complex{Float64},Dim,1)
	@views State[:,1] .= normalize(Para[1:Dim]+1im*Para[Dim+1:2*Dim])
	ProbPl=zeros(2*Dim,It)
	ProbMi=zeros(2*Dim,It)
	#Theoretical (model) log probabilities
	@views @. ProbPl[1:Dim,:].= abs2(State-GuessPl[:,1:It])
	@views @. ProbPl[(Dim +1):end,:].= abs2(State+GuessPl[:,1:It])
	@views @. ProbMi[(Dim +1):end,:].=  abs2(State+GuessMi[:,1:It])
	@views @. ProbMi[1:Dim,:].= abs2(State-GuessMi[:,1:It])

	@. ProbPl = log( .25 *  ProbPl )
	@. ProbMi = log( .25 *  ProbMi )
	#KL divergence (model dependent part)
    return @views-sum(DataPl[:,1:It].*ProbPl+DataMi[:,1:It].*ProbMi)
end

function Graphic(NameArch)
	File=h5open(NameArch,"r")

	(It,Repe,M,N,Ini,Fin,Dim)=File["Info"][:]
	GMB=File["Bound"][:,:]
	Tot=Fin-Ini+1
	Data=zeros(It,Repe,M,N,Tot)

	for i=1:Tot
		Data[:,:,:,:,i]=File["N"*string(i)][:,:,:,:]
	end
	close(File)

x=1:It
MeanRepe=mean(Data,dims=2)[:,1,:,:,:]
MeanGuess=mean(MeanRepe,dims=2)[:,1,:,:]
MeanMSE=mean(MeanGuess,dims=2)[:,1,:]

MedianGuess=median(MeanRepe,dims=2)[:,1,:,:]
MedianMSE=median(MeanGuess,dims=2)[:,1,:]
VarGuess=var(MeanRepe,dims=2)[:,1,:,:]
VarMSE=var(MeanGuess,dims=2)[:,1,:]
MSEPVar=MeanMSE+VarMSE/2
MSEMVar=MeanMSE-VarMSE/2
GuessPVar=MeanGuess+VarGuess/2
GuessMVar=MeanGuess-VarGuess/2


rc("xtick",labelsize=24)
rc("ytick",labelsize=24)

iqrp=zeros(It,Tot) ;iqrm=zeros(It,Tot)
    for l=1:Tot
        for j=1:It
            iqrp[j,l],iqrm[j,l]=quantile(MeanGuess[j,:,l],[.75 .25])
        end
    end

Cols=2
Rows=1
fig,ax=subplots(nrows=Rows,ncols=Cols,figsize=(25,13))
Colors=["deepskyblue" "tomato" "gold" "mediumpurple" "greenyellow"]

Points=[ "1" "o" "^"]

for k=1:3
	ax[1].loglog(x,MeanMSE[:,k],color=Colors[k],label="N=10^"*string(k+2),lw=.3)
	ax[2].loglog(x,MedianMSE[:,k],color=Colors[k],label="N=10^"*string(k+2),lw=.3)
        ax[1].loglog(x,GMB[k,:],color=Colors[k])
        ax[2].loglog(x,GMB[k,:],color=Colors[k])
        ax[1].fill_between(x,MSEPVar[:,k],MSEMVar[:,k],color=Colors[k],alpha=.5)
        ax[2].fill_between(x,iqrp[:,k],iqrm[:,k],color=Colors[k],alpha=.5)
end
    ax[1].grid()
    ax[1].legend(fontsize=24)
    ax[2].grid()
    ax[2].legend(fontsize=24)


fig.text(0.5, 0.01 , "k",ha="center", size=45)
fig.text(0.08,.5,"Inf",ha="center",rotation="vertical", size=45)

fig.savefig("meanmedian"*string(Dim)*".pdf")
clf()

end



main()
