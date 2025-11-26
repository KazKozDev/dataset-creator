import React, { useState, useEffect } from 'react';
import {
    Box,
    Button,
    FormControl,
    FormLabel,
    Input,
    VStack,
    HStack,
    useToast,
    Text,
    Select,
    Heading,
} from '@chakra-ui/react';
import Editor from '@monaco-editor/react';
import axios from 'axios';

const PromptEditor = () => {
    const [templates, setTemplates] = useState([]);
    const [selectedTemplate, setSelectedTemplate] = useState('');
    const [templateName, setTemplateName] = useState('');
    const [editorContent, setEditorContent] = useState('');
    const [testResult, setTestResult] = useState('');
    const toast = useToast();

    useEffect(() => {
        fetchTemplates();
    }, []);

    const fetchTemplates = async () => {
        try {
            const response = await axios.get('http://localhost:8000/api/prompts/templates');
            setTemplates(response.data.templates);
        } catch (error) {
            toast({
                title: 'Error fetching templates',
                description: error.message,
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    const handleTemplateChange = async (e) => {
        const name = e.target.value;
        setSelectedTemplate(name);
        if (name) {
            try {
                const response = await axios.get(`http://localhost:8000/api/prompts/templates/${name}`);
                setTemplateName(response.data.template.metadata.name);
                setEditorContent(JSON.stringify(response.data.template, null, 2));
            } catch (error) {
                console.error(error);
            }
        } else {
            setTemplateName('');
            setEditorContent('');
        }
    };

    const handleSave = async () => {
        try {
            const templateData = JSON.parse(editorContent);
            await axios.post('http://localhost:8000/api/prompts/templates', {
                template_data: templateData,
                overwrite: true
            });
            toast({
                title: 'Template saved',
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
            fetchTemplates();
        } catch (error) {
            toast({
                title: 'Error saving template',
                description: error.message,
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        }
    };

    const handleTest = async () => {
        try {
            const templateData = JSON.parse(editorContent);
            // Mock test for now as we need a specific endpoint for testing raw templates
            // Or we can use the render endpoint if we extract variables
            const response = await axios.post('http://localhost:8000/api/prompts/validate', templateData);
            setTestResult(JSON.stringify(response.data, null, 2));
        } catch (error) {
            setTestResult(error.message);
        }
    };

    return (
        <Box p={5}>
            <Heading size="md" mb={6}>Prompt Templates</Heading>
            <VStack spacing={4} align="stretch">
                <FormControl>
                    <FormLabel>Select Template</FormLabel>
                    <Select placeholder="Select template" value={selectedTemplate} onChange={handleTemplateChange}>
                        {templates.map((t, idx) => {
                            const name = t?.metadata?.name || t?.name || `Template ${idx}`;
                            return (
                                <option key={name} value={name}>
                                    {name}
                                </option>
                            );
                        })}
                    </Select>
                </FormControl>

                <FormControl>
                    <FormLabel>Template Name</FormLabel>
                    <Input value={templateName} onChange={(e) => setTemplateName(e.target.value)} />
                </FormControl>

                <Box h="500px" border="1px solid #ccc">
                    <Editor
                        height="100%"
                        defaultLanguage="json"
                        value={editorContent}
                        onChange={(value) => setEditorContent(value)}
                        options={{ minimap: { enabled: false } }}
                    />
                </Box>

                <HStack>
                    <Button colorScheme="blue" onClick={handleSave}>Save Template</Button>
                    <Button colorScheme="green" onClick={handleTest}>Test Template</Button>
                </HStack>

                {testResult && (
                    <Box p={4} bg="gray.100" borderRadius="md">
                        <Text fontWeight="bold">Test Result:</Text>
                        <pre>{testResult}</pre>
                    </Box>
                )}
            </VStack>
        </Box>
    );
};

export default PromptEditor;
