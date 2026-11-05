%% DMP skill learning: WLR vs GMM-GMR forcing term
% Plain-script version of the code in dmp_gmm_gmr.mlx (the live script also
% contains the write-up and figures). Run from the matlab/ folder.
%
% Sections:
%   1. Original DMP with locally weighted regression (WLR)
%   2. DMP with GMM + GMR forcing term
%   3. Evaluation: number of demonstrations, number of kernels/Gaussians
%   4. Export the learned trajectory for the robot simulation (RobotSimulation.slx)

%% 1. Original DMP (WLR)
nbData = 200;               % Length of each demo trajectory
nbDemos = 5;                % Number of demonstrations for the DMP training
demos = [];
load('Demonstrations.mat');
demos_in = {demos{1:nbDemos}};

nStates = 9;                % Number of activation functions (i.e., number of states in the forcing term)
nVar = 3;                   % Number of variables [s,f1,f2] (decay term and perturbing force)
beta = 50;                  % Stiffness gain (β)
alpha = (2*beta)^.3;        % Damping gain (α)
decayFactor = 1.1;          % Decay factor (a in τx'=-ax)
dt = 0.01;                  % Duration of time step (τ)

dmp = SkillGeneralisation(nVar, nStates, beta, alpha, dt);
dmp = dmp.canonicalSystemInitialisation(decayFactor, nbData);
dmp = dmp.trajectort2Forcing(demos_in);
fWLR = dmp.fittingWithLocallyWLS();
trajWLR = dmp.forcing2Trajectory(fWLR);

figure,
for n = 1:nbDemos
    plot(demos_in{n}.pos(1,:), demos_in{n}.pos(2,:),'.','markersize',8,'color',[.7 .7 .7]); hold on
end
plot(trajWLR(1,:),trajWLR(2,:),'-','linewidth',3,'color',[.8 0 0]);
xlabel('Position X'); ylabel('Position Y');
title("WLR based Trajectory")
axis equal; axis square; hold off

%% 2. DMP with GMM + GMR
dmp = SkillGeneralisation(nVar, nStates, beta, alpha, dt);
dmp = dmp.canonicalSystemInitialisation(decayFactor, nbData);
dmp = dmp.trajectort2Forcing(demos_in);

fGMR = dmp.fittingWithGMR();            % also plots the EM log-likelihood
trajGMR = dmp.forcing2Trajectory(fGMR);

figure,
for n = 1:nbDemos
    plot(demos_in{n}.pos(1,:), demos_in{n}.pos(2,:),'.','markersize',8,'color',[.7 .7 .7]); hold on
end
plot(trajGMR(1,:),trajGMR(2,:),'-','linewidth',3,'color',[0 .7 0]);
xlabel('Position X'); ylabel('Position Y');
title("GMM-GMR based Trajectory")
axis equal; axis square; hold off

%% 3. Evaluation
figure,
for n = 1:nbDemos
    plot(demos_in{n}.pos(1,:), demos_in{n}.pos(2,:),'.','markersize',8,'color',[.7 .7 .7]); hold on
end
plot(trajWLR(1,:),trajWLR(2,:),'-','linewidth',3,'color',[.8 0 0]);
plot(trajGMR(1,:),trajGMR(2,:),'-','linewidth',3,'color',[0 .7 0]);
xlabel('Position X'); ylabel('Position Y');
title("WLR vs GMM-GMR trajectory")
axis equal; axis square; hold off

% RMSE vs number of demonstrations
SkillGeneralisation.compareDemosWLR(demos_in, nbData, beta, alpha, dt, nStates);
SkillGeneralisation.compareDemos(demos_in, nbData, beta, alpha, dt, nStates);

% RMSE vs number of RBF kernels (WLR) / Gaussians (GMM)
kRange = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50];
SkillGeneralisation.compareKvaluesWLR(dmp.dataTraining, demos_in, dmp.xIn, nbData, kRange, ...
    beta, alpha, dt, dmp.endPos, dmp.startPos, dmp.numVarPos, dmp.numDemos);
SkillGeneralisation.compareKvalues(dmp.dataTraining, demos_in, dmp.xIn, nbData, kRange, ...
    beta, alpha, dt, dmp.endPos, dmp.startPos, dmp.numVarPos);

%% 4. Export learned trajectory for the robot simulation
save("LearnedTrajectory.mat", 'trajGMR');
LearnedTrajectory = load("LearnedTrajectory.mat");
X_pos = LearnedTrajectory.trajGMR(1,:)';
Y_pos = LearnedTrajectory.trajGMR(2,:)';
X_offset = 2.5;             % shift the letter into the arm's workspace
Y_offset = 9.5;
retrievedTraj = [X_pos + X_offset, Y_pos + Y_offset];
retrievedTraj_time = linspace(0, 1, size(retrievedTraj, 1))';   % 1 s trajectory
dataset = timeseries(retrievedTraj, retrievedTraj_time, 'Name', 'SimData');
save("SimData.mat", "dataset", "-v7.3");
% Then open RobotSimulation.slx and run it.
