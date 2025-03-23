FROM node:16-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy application source code
COPY frontend/ .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]