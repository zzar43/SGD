# acoustic_solver: 2d FD solver for acoustic wave equation
# acoustic_solver_all: 2d FD solver for acoustic wave equation for each source one by one, used for seismic forward modelling

# The base function for acoustic equation solver
# a = 1/(c^2 rho), b = 1/rho
# source_position and receiver_position dimension: (number of sources or receivers) \times 2
# source_func dimension: Nt \times (number of sources)
# pml_len: lenghth of the PML layer
# pml_alpha: coefficient of the PML damping function
# source_index = 0: all sources activating together
# source_index = 1: only the first source is activated
function acoustic_solver(a, b, Nx, Ny, Nt, dx, dy, dt, source_position, source_func, receiver_position; pml_len=30, pml_alpha=50, source_index=0)

    if size(source_position, 1) != size(source_func, 2)
        error("Please check the size of source_func and source_position.")
    end
    if size(source_position, 2) != 2
        error("Please check the dimension of source_position.")
    elseif size(receiver_position, 2) != 2
        error("Please check the dimension of receiver_position.")
    end

    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
        
    if typeof(source_position) == Array{Int64,1}
        source_temp = zeros(1,2);
        source_temp[1,1] = source_position[1];
        source_temp[1,2] = source_position[2];
        source_position = copy(source_temp);
    end
    if source_index ==0
        source_position_pml = source_position .+ pml_len;
    else
        source_position_pml = zeros(Int, 1,2);
        source_position_pml[1,1] = source_position[source_index,1] + pml_len
        source_position_pml[1,2] = source_position[source_index,2] + pml_len

        source_func = source_func[:,source_index]
    end
    receiver_position_pml = receiver_position .+ pml_len;
    
    pml_coef = range(0,stop=pml_alpha,length=pml_len);
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    for i = 1:pml_len
        sigma_x[pml_len-i+1,:] .= pml_coef[i];
        sigma_x[end-pml_len+i,:] .= pml_coef[i];
        sigma_y[:,pml_len-i+1] .= pml_coef[i];
        sigma_y[:,end-pml_len+i] .= pml_coef[i];
    end
    
    a = extend_vel(a, Nx_pml, Ny_pml, pml_len);
    b = extend_vel(b, Nx_pml, Ny_pml, pml_len);

    u = zeros(Nx_pml, Ny_pml);
    u0 = zeros(Nx_pml, Ny_pml);

    vx = zeros(Nx_pml, Ny_pml);
    vx0 = zeros(Nx_pml, Ny_pml);

    vy = zeros(Nx_pml, Ny_pml);
    vy0 = zeros(Nx_pml, Ny_pml);

    psi = zeros(Nx_pml, Ny_pml);
    psi0 = zeros(Nx_pml, Ny_pml);

    phi = zeros(Nx_pml, Ny_pml);
    phi0 = zeros(Nx_pml, Ny_pml);
    
    data = zeros(Nt, size(receiver_position,1))
    wavefield = zeros(Nx, Ny, Nt)
        
    for iter = 1:Nt
        temp = b[2:end-1, 2:end-1]  .* (forward_diff_x(vx, dx) .+ forward_diff_y(vy,dy));
        u[2:end-1, 2:end-1] = u0[2:end-1, 2:end-1] .+ dt .* (temp - (sigma_x[2:end-1, 2:end-1] .+ sigma_y[2:end-1, 2:end-1]).*u[2:end-1, 2:end-1] + psi[2:end-1, 2:end-1] + phi[2:end-1, 2:end-1]);
        u[(source_position_pml[:,2] .- 1) * Nx_pml .+ source_position_pml[:,1]] .= source_func[iter,:];

        vx[2:end-1, 2:end-1] = vx0[2:end-1, 2:end-1] + dt .* (a[2:end-1, 2:end-1] .* backward_diff_x(u,dx) - sigma_x[2:end-1, 2:end-1] .* vx0[2:end-1, 2:end-1]);
        vy[2:end-1, 2:end-1] = vy0[2:end-1, 2:end-1] + dt .* (a[2:end-1, 2:end-1] .* backward_diff_y(u,dy) - sigma_y[2:end-1, 2:end-1] .* vy0[2:end-1, 2:end-1]);

        psi[2:end-1, 2:end-1] = psi0[2:end-1, 2:end-1] + dt .* (b[2:end-1, 2:end-1] .* sigma_x[2:end-1, 2:end-1] .* backward_diff_y(vy,dy));
        phi[2:end-1, 2:end-1] = phi0[2:end-1, 2:end-1] + dt .* (b[2:end-1, 2:end-1] .* sigma_y[2:end-1, 2:end-1] .* backward_diff_x(vx,dx));

        u0 = copy(u)
        vx0 = copy(vx)
        vy0 = copy(vy)
        psi0 = copy(psi)
        phi0 = copy(phi)
        
        data[iter,:] .= u[(receiver_position_pml[:,2] .- 1) * Nx_pml .+ receiver_position_pml[:,1]]
        wavefield[:,:,iter] .= u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
    end
    
    return data, wavefield
end

# Solve the aoucstic equation with all sources one by one
function acoustic_solver_all(a, b, Nx, Ny, Nt, dx, dy, dt, source_position, source_func, receiver_position; pml_len=30, pml_alpha=50)

    Ns = size(source_position,1)
    received_data = zeros(Nt, size(receiver_position,1), Ns)
    wavefield = zeros(Nx, Ny, Nt, Ns)

    for iter = 1:Ns
        data, u = acoustic_solver(a, b, Nx, Ny, Nt, dx, dy, dt, source_position, source_func, receiver_position; pml_len=pml_len, pml_alpha=pml_alpha, source_index=iter)
        received_data[:,:,iter] = data;
        wavefield[:,:,:,iter] = u;
    end
    
    return received_data, wavefield
end

# Parallel version of acoustic_solver_all. If don't need, please comment it
using Distributed
using SharedArrays
function acoustic_solver_all_parallel(a, b, Nx, Ny, Nt, dx, dy, dt, source_position, source_func, receiver_position; pml_len=30, pml_alpha=50)

    Ns = size(source_position,1)
    received_data = SharedArray{Float64}(Nt, size(receiver_position,1), Ns);
    wavefield = SharedArray{Float64}(Nx, Ny, Nt, Ns);
    
    @sync @distributed for iter = 1:Ns
        data, u = acoustic_solver(a, b, Nx, Ny, Nt, dx, dy, dt, source_position, source_func, receiver_position; pml_len=pml_len, pml_alpha=pml_alpha, source_index=iter)
        received_data[:,:,iter] = data;
        wavefield[:,:,:,iter] = u;
    end
    
    return received_data, wavefield
end

# Ricker function for source
function source_ricker(center_fre, center_time, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t.-center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2);
end

function extend_vel(vel, Nx_pml, Ny_pml, pml_len)
    vel_ex = zeros(Nx_pml, Ny_pml);
    vel_ex[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .= vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end
    return vel_ex
end

forward_diff_x(A, dx) = (A[3:end,2:end-1] - A[2:end-1,2:end-1])/dx;
backward_diff_x(A, dx) = (A[2:end-1,2:end-1] - A[1:end-2,2:end-1])/dx;
forward_diff_y(A, dy) = (A[2:end-1,3:end] - A[2:end-1,2:end-1])/dy;
backward_diff_y(A, dy) = (A[2:end-1,2:end-1] - A[2:end-1,1:end-2])/dy;