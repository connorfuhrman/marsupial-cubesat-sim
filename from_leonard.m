%
%  draws trajectories of emitted particles from Bennu using velocity
%  derived from observations, choosing a random longitude, but a latitude
%  proportional to the cosine of the latitude as derived in dissertation.
%
%  L.D. Vance 17 October 2022
%  
rng(1);         %  set random number seed
nruns = 50;     %  number of trajectories to model
dt = 5;         %  integration time step;
tmax = 20*3600;  %  max time simulation is allowed to run
w_bennu = (2*pi)/(4.30*3600)*[0 0 1]';  %  angular rate of bennu, Detection of 
                                        %  Rotational Acceleration of Bennu
                                        %  Using HST Light Curve Observations},  
                                        %  M.C. Nolan et al, Geophysical Research Letters,  2018
mu_bennu = 4.9;                         % m3/s2  from Scheere's Nature paper
%
t_out = zeros(1,nruns);
istop = zeros(1,nruns);
Rx_out = zeros(nruns,100000);
Ry_out = zeros(nruns,100000);
Rz_out = zeros(nruns,100000);
%
%  create cosine latitude related latitude random set
%
nps = 1000;  %  number of points in table lookup for cosine(lat) effect
lat_tab = [0:pi/(2*nps):pi/2];
%
%  form table of integ of cos2 for latitude launch probability lookup
%
integ_cos2_table = 0.5*lat_tab + 0.25*sin(2*lat_tab);

inp_draw = rand*pi/4;
lat_launch = sign(randn)*interp1(integ_cos2_table,lat_tab,inp_draw);
long_launch = 2*pi*(rand-0.5);

for irun=1:nruns
    %
    %  setup initial conditions
    %
    rad_launch = 233;  %  from 40 degree chart of actual Bennu  interp1(lat,rad1_bennu(2:nlats+1),lat_launch);
    R = rad_launch*[cos(long_launch)*cos(lat_launch);sin(long_launch)*cos(lat_launch);sin(lat_launch)];
    R_launch = R;
    Rhat = R/norm(R);
    %
    %  velocity
    %
    vmag_mean = 0.095;
    V_surface = skew(w_bennu)*R;  %  velocity of rock lying on surface
    v0_mag = vmag_mean*(1/1.5958)*norm(randn(3,1));  %  1.5958 from maxwell distribution function
    %
    %  choose a velocity vector within a specified angle from vertical
    %
    V3hat_rand = unit(randn(3,1));
    while acos(V3hat_rand'*Rhat)>pi/2
     V3hat_rand = unit(randn(3,1));
    end       
    V_eject = v0_mag*V3hat_rand;
    V = V_eject + V_surface;
    %
    %  integration time loop
    %
    time = 0;
    itime = 0;
    %
    %  run sim if trajectory is suborbital
    %
    %rad_lat = interp1(lat,rad1_bennu(2:nlats+1),lat_launch);
    altitude = 1.0e-3+ norm(R)-rad_launch;
    while (altitude>=0 || R'*V>0) && time<tmax
        itime = itime+1;
        time = time + dt;
        %
        psi_launch = w_bennu(3)*time + long_launch;
        %
        %  half step position propagation
        %
        R_half = R + 0.5*dt*V;
        %
        %  calculate accelerations in Neptune centric coordinate frame.
        %  Store old acceleration values for comparison with newer value
        %  regarding time step change
        %
        if itime >=2
            A_half_old = A_half;
        else
            A_half_old = [0 0 0]';
        end
        %
        %  acceleration
        %
        A_half = -mu_bennu*R_half /norm( R_half)^3;  %  bennu attractive gravity
        %
        %  integrate accelerations into velocities
        %
        V = V + dt*A_half;
        %
        %  integrate velocities into positions
        %
        R = R_half + 0.5*dt*V;
        %
        %   output trajectories
        %
        r_mag = norm(R);
%         latitude = atan2(R(3),norm(R(1:2)));
%         rad_lat = interp1(lat,rad1_bennu(2:nlats+1),latitude);
        altitude = r_mag - rad_launch;

        Rx_out(irun,itime) = R(1);
        Ry_out(irun,itime) = R(2);
        Rz_out(irun,itime) = R(3);
    end  %  time index
    istop(irun)  = itime;
    
end % number of simulation runs

%
%  Load NASA Bennu facet model
%
model = stlread("Bennu_v20_200k.stl");
%
%  plot Bennu 3d model
%
figure(1);clf;
colordef(1,'black');
trisurf(model,'FaceColor',[0.85 0.8 0.65],'EdgeColor','none','FaceAlpha',1.0);
axis equal
light('Color',[1 1 1],'Position',[1 -0.3 0.3]);
material dull; hold on
%
%  outlines of profile evolution
%
% plot3(yp,xp ,zp  ,'Color',[0.3 0.3 1.0],'LineWidth',1);
% plot3(ypa,xpa,zpa,'Color',[0.7 0.3 1.0],'LineWidth',1);
% plot3(ypb,xpb,zpb,'Color',[1.0 0.3 0.7],'LineWidth',1);
% plot3(yp1,xp1,zp1,'Color',[1.0 0.3 0.0],'LineWidth',1);
%
%  trajectories from bennu_cartoon
%
for irun = 1:nruns
    plot3(Rx_out(irun,1:istop(irun))/1000,Ry_out(irun,1:istop(irun))/1000,Rz_out(irun,1:istop(irun))/1000,'Color',[0.9 0.9 0.9],'LineWidth',1);
end
%
%
%  rotation axis
%
plot3([0 0],[0 0],[-.35 .35],'r');
view([1 0 0]);
axis(1.0*[-1 1 -1 1 -1 1]);
xlabel('x (km)');
ylabel('y (km)');
zlabel('z (km)');


