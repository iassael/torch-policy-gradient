RK = require 'Math.RungeKutta'
require 'leaf.polygon'


function dynamics_sp(t, z)

    -- z 1,2 ang velocity, 3,4 angles
    
    local m = 1  -- [kg]     mass of 1st link
    local b = 0.1  -- [Ns/m]   coefficient of friction (1st joint)
    local l = 1  -- [m]      length of 1st pendulum
    local g = 9.82 -- [m/s^2]  acceleration of gravity
    -- local g = 0  -- [m/s^2]  acceleration of gravity


    dz = {}
    dz[1] = ( f1 - b*z[1] - m*g*l*math.sin(z[2])/2 ) / (m*l^2/3)
    dz[2] = z[1]
    
    
    -- return dz
    return dz
end

function dynamics_dp(t, z)

	-- z 1,2 ang velocity, 3,4 angles
    
    local m1 = 1  -- [kg]     mass of 1st link
    local m2 = 1  -- [kg]     mass of 2nd link
    local b1 = 0.1  -- [Ns/m]   coefficient of friction (1st joint)
    local b2 = 0.1  -- [Ns/m]   coefficient of friction (2nd joint)
    local l1 = 1  -- [m]      length of 1st pendulum
    local l2 = 1  -- [m]      length of 2nd pendulum
    local g = 9.82 -- [m/s^2]  acceleration of gravity
    -- local g = 0 -- [m/s^2]  acceleration of gravity
    local I1 = m1*l1^2/12  -- moment of inertia around pendulum midpoint (1st link)
    local I2 = m2*l2^2/12  -- moment of inertia around pendulum midpoint (2nd link)
    
    local A = torch.Tensor({{l1^2*(0.25*m1+m2) + I1,  0.5*m2*l1*l2*math.cos(z[3]-z[4])},
                       {0.5*m2*l1*l2*math.cos(z[3]-z[4]), l2^2*0.25*m2 + I2}})
    
    local b = torch.Tensor({{g*l1*math.sin(z[3])*(0.5*m1+m2) - 0.5*m2*l1*l2*z[2]^2*math.sin(z[3]-z[4]) + f1-b1*z[1]},
                       {0.5*m2*l2*(l1*z[1]^2*math.sin(z[3]-z[4])+g*math.sin(z[4])) + f2-b2*z[2]}})
    --x = A\b
    local x = torch.gesv(b, A):view(2)
    
    -- return dz
    return {x[1], x[2], z[1], z[2]}
end

function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end


function control_sp(y0, dt, f_steps)
    local t = 0 y = tools.dc(y0)
    
    local states = {}
    
    for t = 1,#f_steps do
        -- next step
        f1 = f_steps[t][1]
        _, y = RK.rk4(y, dynamics_sp, t, dt)  -- Merson's 4th-order method
        states[#states+1] = tools.dc(y)
    end
    
    return states
end


function control_dp(y0, dt, f_steps)
    local t = 0 y = tools.dc(y0)
    
    local states = {}
    
    for t = 1,#f_steps do
        -- next step
        f1 = f_steps[t][1]
        f2 = f_steps[t][2]
        _, y = RK.rk4(y, dynamics_dp, t, dt)  -- Merson's 4th-order method
        states[#states+1] = y
    end
    
    return states
end


function draw_rect(frame, ul, ur, bl, br)
    local poly = leaf.Polygon()
    poly:addPoint(ul[2],ul[1])
    poly:addPoint(bl[2],bl[1])
    poly:addPoint(br[2],br[1])
    poly:addPoint(ur[2],ur[1])

    local min_x = math.min(ul[2],bl[2],br[2],ur[2])
    local max_x = math.max(ul[2],bl[2],br[2],ur[2])

    local min_y = math.min(ul[1],bl[1],br[1],ur[1])
    local max_y = math.max(ul[1],bl[1],br[1],ur[1])
    
    
    for x=min_x,max_x do
        for y=min_y,max_y do
            if poly:contains(x,y) then
                frame[{{y},{x}}] = 1
            end
        end
    end
end

function draw_dp(t1, t2, img_size)
    
    local img_size
    if img_size == nil then
        img_size = 48
    end

    -- screen px
    local scale = 4
    local nx = img_size * scale
    local ny = nx

    -- radius (in pixels) of each bob (min/max: 3/12 pixels)
    local r1 = 3 * scale
    local r2 = r1

    -- length (in pixels) of each rod
    local l1 = 1  -- [m]      length of 1st pendulum
    local l2 = 1  -- [m]      length of 2nd pendulum
    local p1 = 0.85 * math.min(nx/2,ny/2) * (l1 / (l1 + l2))
    local p2 = 0.85 * math.min(nx/2,ny/2) * (l2 / (l1 + l2))

    -- positions (in (pixels,pixels)) of each bob
    x0 = torch.zeros(2)
    x1 = x0 + torch.Tensor({p1*math.sin(t1), p1*math.cos(t1)})
    x2 = x1 + torch.Tensor({p2*math.sin(t2), p2*math.cos(t2)})

    -- Define rectangle centers
    center_l1 = (x0 + x1) / 2 + nx / 2
    center_l2 = (x1 + x2) / 2 + ny / 2

    -- Pendulum 2 points
    local width = p1
    local height = r1
    local angle = math.atan2(x0[1]-x1[1], x0[2]-x1[2])
    local cos_angle = math.cos(angle)
    local sin_angle = math.sin(angle)
    local center = center_l1
    BR1 = {(center[2] + (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  + (width/2) * sin_angle)}
    UR1 = {(center[2] - (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  - (width/2) * sin_angle)}
    BL1 = {(center[2] + (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  + (width/2) * sin_angle)}
    UL1 = {(center[2] - (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  - (width/2) * sin_angle)}

    -- Pendulum 2 points
    width = p2
    height = r2
    angle = math.atan2(x1[1]-x2[1], x1[2]-x2[2])
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    center = center_l2
    BR2 = {(center[2] + (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  + (width/2) * sin_angle)}
    UR2 = {(center[2] - (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  - (width/2) * sin_angle)}
    BL2 = {(center[2] + (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  + (width/2) * sin_angle)}
    UL2 = {(center[2] - (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  - (width/2) * sin_angle)}


    -- Create frame
    local frame = torch.zeros(ny,nx)
    -- frame[{UL1[1], UL1[2]}] = 1
    -- frame[{UR1[1], UR1[2]}] = 1
    -- frame[{BL1[1], BL1[2]}] = 1
    -- frame[{BR1[1], BR1[2]}] = 1

    -- frame[{UL2[1], UL2[2]}] = 1
    -- frame[{UR2[1], UR2[2]}] = 1
    -- frame[{BL2[1], BL2[2]}] = 1
    -- frame[{BR2[1], BR2[2]}] = 1

    -- Draw pendulum
    draw_rect(frame, UL1, UR1, BL1, BR1)
    draw_rect(frame, UL2, UR2, BL2, BR2)

    -- scale down
    frame = image.scale(frame, img_size)

    return frame
end


function draw_sp(t1, img_size, scale)
    
    if img_size == nil then
        img_size = 48
    end

    -- screen px
    local scale = scale or 4
    local nx = img_size * scale
    local ny = nx

    -- radius (in pixels) of each bob (min/max: 3/12 pixels)
    local r1 = 3 * scale

    -- length (in pixels) of each rod
    local l1 = 1  -- [m]      length of 1st pendulum
    local p1 = 0.85 * math.min(nx/2,ny/2) --* (l1 / (l1 + l2))

    -- positions (in (pixels,pixels)) of each bob
    x0 = {0, 0}
    x1 = {p1*math.sin(t1), p1*math.cos(t1)}

    -- Define rectangle centers
    center_l1 = {}
    center_l1[1] = (x0[1] + x1[1]) / 2 + nx / 2
    center_l1[2] = (x0[2] + x1[2]) / 2 + nx / 2

    -- Pendulum 2 points
    local width = p1
    local height = r1
    local angle = math.atan2(x0[1]-x1[1], x0[2]-x1[2])
    local cos_angle = math.cos(angle)
    local sin_angle = math.sin(angle)
    local center = center_l1
    BR1 = {(center[2] + (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  + (width/2) * sin_angle)}
    UR1 = {(center[2] - (width/2) * cos_angle - (height/2) * sin_angle), (center[1] + (height/2) * cos_angle  - (width/2) * sin_angle)}
    BL1 = {(center[2] + (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  + (width/2) * sin_angle)}
    UL1 = {(center[2] - (width/2) * cos_angle + (height/2) * sin_angle), (center[1] - (height/2) * cos_angle  - (width/2) * sin_angle)}


    -- Create frame
    local frame = torch.zeros(ny, nx)

    -- Draw pendulum
    draw_rect(frame, UL1, UR1, BL1, BR1)

    -- scale down
    if scale > 1 then
        frame = image.scale(frame, img_size)
    end

    return frame
end


function generate_dataset(steps, dt, img_size)
	local z_start = {0,0,0,0} -- 1,2 ang. vel., 3,4 angle

	local z = {[1] = z_start}
	local u = torch.randn(steps, 2)

	local x = torch.zeros(steps+1, img_size^2)
	x[1] = draw_pendulum(z[1][3], z[1][4], img_size):view(img_size^2)

	for t=1,steps do
	    -- Make a step
	    z[t+1] = control_dp(z[t], dt, {u[t][1]}, {u[t][2]})[1]
	    
	    -- Display image
	    x[t+1] = draw_pendulum(z[t+1][3], z[t+1][4], img_size):view(img_size^2)
	    -- disp_img(draw_pendulum(z[t][3], z[t][4]))
	end

	return z, x, u
end
