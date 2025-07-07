# Use official Nginx image to serve static frontend
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy frontend files (public) to nginx html directory
COPY public /usr/share/nginx/html

# Expose port 80
EXPOSE 80
